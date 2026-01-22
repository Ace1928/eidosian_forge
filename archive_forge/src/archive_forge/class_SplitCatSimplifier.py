import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
class SplitCatSimplifier:
    """
    Helper class to simplify split-cat pattern. In simple cases, both split and cat node can be removed in a "split->cat"
    pattern. However, there are various cases where they can't and we need to simplify split/ add transforms before cat.
    Some such cases are:
        1. Final node has additional args (not coming from the initial split)
        2. Shuffling of args between split/cat
        3. Some final nodes are non-(cat/stack)
        4. Split-dim != cat-dim (but equal split)

    Note that any combination of the above cases can happen.

    To deal with 1, 2, & 3 - we iterate over all users of split. And figure out common "ranges" that can be merged.
    Then, we simplify the split accordingly. In the best case, split can be entirely removed.

    To deal with 4, we add some transformations (unflatten + movedim) (See `get_transform_params`).

    Finally, depending on final node being cat or stack, unsqueeze/flatten needs to be added.

    """

    def simplify(self, graph: torch.fx.Graph, split_node: torch.fx.Node, split_sections: List[int]):
        next_users = find_next_users(split_node)
        user_inputs_list = self.get_user_input_list(split_node, next_users)
        simplified_split_ranges = self.get_simplified_split_ranges(split_sections, next_users, user_inputs_list)
        if not simplified_split_ranges:
            return
        transform_params_list = self.get_transform_params(split_node, next_users, user_inputs_list)
        if not transform_params_list:
            return
        user_inputs_list_new = self.replace_split(graph, split_node, split_sections, user_inputs_list, simplified_split_ranges)
        self.replace_cat(graph, split_node, next_users, user_inputs_list_new, transform_params_list)
        self.erase_old_nodes(graph, split_node, next_users)

    def get_user_input_list(self, split_node: torch.fx.Node, next_users: List[torch.fx.Node]) -> List[List[Union[torch.fx.Node, _Range]]]:
        """
        Returns list of inputs to the following user nodes, in order. The outer list represents the user node. The inner
        list represents the inputs to that particular node. This list can either contain
          - a tuple representing the ranges of get_items that should go into the cat (closed interval)
          - torch.fx.Node representing "other" inputs (which are not coming from our split)
        """
        user_inputs_list: List[List[Union[torch.fx.Node, _Range]]] = []
        for user in next_users:
            if user.target in {torch.cat, torch.stack}:
                user_inputs_list.append(self.get_merged_user_inputs(split_node, user))
            else:
                user_inputs_list.append(self.get_non_cat_node_input(split_node, user))
        return user_inputs_list

    def get_merged_user_inputs(self, split_node: torch.fx.Node, cat_node: torch.fx.Node) -> List[Union[torch.fx.Node, _Range]]:
        user_inputs = get_arg_value(cat_node, 0, 'tensors')
        simplified_user_inputs = []
        split_users = set(split_node.users.keys())
        for user_input in user_inputs:
            if user_input not in split_users:
                simplified_user_inputs.append(user_input)
            else:
                simplified_user_inputs.append(user_input.args[1])
        return self.merge_consecutive_inputs(simplified_user_inputs)

    def get_non_cat_node_input(self, split_node: torch.fx.Node, node: torch.fx.Node) -> List[_Range]:
        """
        Get input for a non cat node in the same format as `get_merged_user_inputs`
        """
        node_input = []
        split_users = set(split_node.users.keys())
        for node_arg in node.all_input_nodes:
            if node_arg in split_users:
                getitem_num = get_arg_value(node_arg, 1)
                node_input.append((getitem_num, getitem_num))
        return node_input

    def merge_consecutive_inputs(self, inputs: List[Union[torch.fx.Node, int]]) -> List[Union[torch.fx.Node, _Range]]:
        """
        Merge consecutive inputs going into a user node.

        For e.g.
        [arg0, 0, 1, 2, arg1] -> [arg0, (0, 2), arg1]
        """
        merged_ranges = []
        cur_range = None
        for input_ in inputs:
            if isinstance(input_, int):
                if not cur_range:
                    cur_range = [input_, input_]
                elif input_ == cur_range[1] + 1:
                    cur_range[1] += 1
                else:
                    merged_ranges.append(tuple(cur_range))
                    cur_range = [input_, input_]
            else:
                if cur_range:
                    merged_ranges.append(tuple(cur_range))
                    cur_range = None
                merged_ranges.append(input_)
        if cur_range:
            merged_ranges.append(tuple(cur_range))
        return merged_ranges

    def get_simplified_split_ranges(self, split_sections, next_users, user_inputs_list: List[List[Union[torch.fx.Node, _Range]]]) -> Optional[List[_Range]]:
        ranges = set()
        for user_node, user_inputs in zip(next_users, user_inputs_list):
            ranges |= {user_input for user_input in user_inputs if isinstance(user_input, tuple)}
        cumulative_sizes = [0] + torch.cumsum(torch.tensor(split_sections), 0).tolist()
        split_ranges = sorted([(cumulative_sizes[r[0]], cumulative_sizes[r[1] + 1]) for r in ranges])
        if not self.has_non_overlapping_ranges(split_ranges):
            return None
        split_ranges = self.fill_gaps(split_ranges, 0, cumulative_sizes[-1])
        if len(split_sections) == len(split_ranges):
            return None
        counters['inductor']['scmerge_split_sections_removed'] = len(split_sections) - len(split_ranges)
        return split_ranges

    def has_non_overlapping_ranges(self, ranges: List[_Range]) -> bool:
        for range_, next_range in zip(ranges, ranges[1:]):
            if range_[1] > next_range[0]:
                return False
        return True

    def fill_gaps(self, ranges: List[_Range], min_: int, max_: int) -> List[_Range]:
        cur = min_
        filled_ranges = []
        for a, b in ranges:
            if cur < a:
                filled_ranges.append((cur, a))
            filled_ranges.append((a, b))
            cur = b
        if filled_ranges[-1][1] < max_:
            filled_ranges.append((filled_ranges[-1][1], max_))
        return filled_ranges

    def get_transform_params(self, split_node: torch.fx.Node, next_users: List[torch.fx.Node], user_inputs_list: List[List[Union[torch.fx.Node, _Range]]]) -> Optional[List[List[_TransformParam]]]:
        """
        Figure out what transforms are needed for each input to each cat node.

        We replace a split node with an unflatten followed by a movedim
        """
        split_dim = split_node.kwargs['dim']
        split_sections = split_node.args[1]
        transform_params_list: List[List[_TransformParam]] = []
        for user_node, user_inputs in zip(next_users, user_inputs_list):
            if user_node.target not in {torch.cat, torch.stack}:
                transform_params_list.append([])
                continue
            cat_dim = get_arg_value(user_node, 1, 'dim')
            transform_params: List[_TransformParam] = []
            for user_input in user_inputs:
                if split_dim == cat_dim and user_node.target == torch.cat:
                    transform_params.append((None, None, None, None))
                elif isinstance(user_input, tuple):
                    subset_split_sections = split_sections[user_input[0]:user_input[1] + 1]
                    if len(set(subset_split_sections)) != 1:
                        return None
                    num_splits = len(subset_split_sections)
                    unflatten_params = (split_dim, (num_splits, -1))
                    movedim_params = (split_dim, cat_dim) if split_dim != cat_dim else None
                    transform_params.append((unflatten_params, movedim_params, None, None))
                elif user_node.target == torch.stack or split_dim != cat_dim:
                    transform_params.append((None, None, (cat_dim,), None))
                else:
                    transform_params.append((None, None, None, None))
            transform_params_list.append(transform_params)
        return transform_params_list

    def replace_split(self, graph: torch.fx.Graph, split_node: torch.fx.Node, split_sections: List[int], user_inputs_list: List[List[Union[torch.fx.Node, _Range]]], split_ranges: List[_Range]) -> List[List[torch.fx.Node]]:
        """
        Replace the split node. It can either remove the split node if len(split_ranges) == 1, or simplify it
        into a split with lesser sections if len(split_ranges) > 1.

        Returns the new `user_inputs_list`, with tuples replaced with new getitems from the newer split node.
        """
        split_input = split_node.args[0]
        split_dim = split_node.kwargs['dim']
        if len(split_ranges) == 1:
            split_items = [split_input]
        else:
            with graph.inserting_after(split_node):
                new_split = graph.call_function(torch.split, args=(split_input, [r[1] - r[0] for r in split_ranges]), kwargs={'dim': split_dim})
                new_split.meta.update(split_node.meta)
                counters['inductor']['scmerge_split_added'] += 1
            with graph.inserting_after(new_split):
                split_items = [graph.call_function(operator.getitem, args=(new_split, i)) for i in range(len(split_ranges))]
        cumulative_sizes = [0] + torch.cumsum(torch.tensor(split_sections), 0).tolist()
        new_user_inputs_list = []
        for user_inputs in user_inputs_list:
            new_user_inputs = []
            for user_input in user_inputs:
                if isinstance(user_input, tuple):
                    new_user_inputs.append(split_items[split_ranges.index((cumulative_sizes[user_input[0]], cumulative_sizes[user_input[1] + 1]))])
                else:
                    new_user_inputs.append(user_input)
            new_user_inputs_list.append(new_user_inputs)
        return new_user_inputs_list

    def replace_cat(self, graph: torch.fx.GraphModule, split_node: torch.fx.Node, next_users: List[torch.fx.Node], user_inputs_list_new, transform_params_list: List[List[_TransformParam]]):
        split_dim = split_node.kwargs['dim']
        split_users = split_node.users.keys()
        new_cats = []
        for user_node, user_inputs_new, transform_params in zip(next_users, user_inputs_list_new, transform_params_list):
            if user_node.target not in {torch.cat, torch.stack}:
                next_cat_input = 0
                for input_node in user_node.all_input_nodes:
                    if input_node in split_users:
                        user_node.replace_input_with(input_node, user_inputs_new[next_cat_input])
                        next_cat_input += 1
                continue
            cat_dim = get_arg_value(user_node, 1, 'dim')
            user_inputs_new_transformed = []
            to_stack = []
            stack_dim = None
            with graph.inserting_before(user_node):
                for user_input_new, transform_param in zip(user_inputs_new, transform_params):
                    unflatten_params, movedim_params, unsqueeze_params, flatten_params = transform_param
                    if unsqueeze_params and (stack_dim is None or stack_dim == unsqueeze_params[0]):
                        to_stack.append(user_input_new)
                        stack_dim = unsqueeze_params[0]
                        continue
                    elif to_stack:
                        stacked_input = graph.call_function(torch.stack, args=(to_stack,), kwargs={'dim': stack_dim})
                        to_stack = []
                        stack_dim = None
                        user_inputs_new_transformed.append(stacked_input)
                        if unsqueeze_params:
                            to_stack.append(user_input_new)
                            stack_dim = unsqueeze_params[0]
                            continue
                    if unflatten_params:
                        user_input_new = graph.call_function(torch.unflatten, args=(user_input_new, *unflatten_params))
                    if movedim_params:
                        user_input_new = graph.call_function(torch.movedim, args=(user_input_new, *movedim_params))
                    if flatten_params:
                        user_input_new = graph.call_function(torch.flatten, args=(user_input_new, *flatten_params))
                    user_inputs_new_transformed.append(user_input_new)
                if to_stack:
                    stacked_input = graph.call_function(torch.stack, args=(to_stack,), kwargs={'dim': stack_dim})
                    user_inputs_new_transformed.append(stacked_input)
            with graph.inserting_after(user_node):
                if len(user_inputs_new_transformed) > 1:
                    new_cat_node = graph.call_function(torch.cat, args=(user_inputs_new_transformed,), kwargs={'dim': cat_dim})
                    new_cat_node.meta.update(user_node.meta)
                    counters['inductor']['scmerge_cat_added'] += 1
                else:
                    new_cat_node = user_inputs_new_transformed[-1]
            if user_node.target == torch.cat and split_dim != cat_dim and (split_node.target == torch.split):
                with graph.inserting_after(new_cat_node):
                    new_cat_node = graph.call_function(torch.flatten, args=(new_cat_node, cat_dim, cat_dim + 1))
            user_node.replace_all_uses_with(new_cat_node)
            new_cats.append(new_cat_node)

    def erase_old_nodes(self, graph: torch.fx.GraphModule, split_node: torch.fx.Node, next_users: List[torch.fx.Node]):
        to_remove = [split_node]
        counters['inductor']['scmerge_split_removed'] += 1
        for getitem_node in split_node.users.keys():
            to_remove.append(getitem_node)
        for next_user in next_users:
            if next_user.target not in {torch.cat, torch.stack}:
                continue
            counters['inductor']['scmerge_cat_removed'] += 1
            to_remove.append(next_user)
        for node in reversed(to_remove):
            graph.erase_node(node)