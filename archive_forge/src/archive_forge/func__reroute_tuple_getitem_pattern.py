import copy
import torch
import torch.nn as nn
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule, map_arg
from torch.fx.graph import (
from .custom_config import PrepareCustomConfig
from ._decomposed import quantized_decomposed_lib  # noqa: F401
from typing import Callable, Optional, List, Dict, Any, Set, Tuple, Union, Type
from dataclasses import dataclass
from collections import namedtuple
import operator
import warnings
def _reroute_tuple_getitem_pattern(graph: Graph):
    """
    Search for patterns where N consecutive `tuple` call_function nodes are followed by
    N consecutive `getitem` call_function nodes that are "reverses" of the `tuple` nodes.
    If we find this pattern, reroute the consumers of the last `getitem` to skip these
    N `tuple` and `getitem` nodes.

    Before:

        a   b     c
        |   \\   /
        \\   tuple
         \\   /
          tuple
            |
        getitem(1)
            |
        getitem(0)
            |
            d

    After:

        b
        |
        d
    """

    def find_patterns(node: Node, index_stack: List[int], current_pattern: List[Node], matched_patterns: List[List[Node]], seen: Set[Tuple[Node, Tuple[int, ...]]]):
        """
        Traverse the graph recursively to match for the N-tuple - N-getitem patterns,
        starting at the given node.

        We use a stack to keep track of the expected `getitem` indices, since these are
        reversed from the `tuple` indices. In the above example, the stack after
        (b -> tuple -> tuple) will be [0, 1], which will be popped by getitem(1) first
        and then by getitem(0).

        TODO: traverse upwards from the output and handle the case when tuple is not a
        separate node, e.g. graph.call_function(operator.getitem, args=(a, (b, c)))
        """
        if len(index_stack) == 0 and len(current_pattern) > 0:
            matched_patterns.append(copy.copy(current_pattern))
            current_pattern.clear()
        state = (node, tuple(index_stack))
        if state in seen:
            return
        seen.add(state)
        for user in node.users:
            if user.op == 'call_function' and user.target == tuple:
                for i, user_arg in enumerate(user.args[0]):
                    if user_arg == node:
                        index_stack.append(i)
                        current_pattern.append(user)
                        find_patterns(user, index_stack, current_pattern, matched_patterns, seen)
            elif user.op == 'call_function' and user.target == operator.getitem:
                if len(index_stack) > 0:
                    if user.args[1] == index_stack[-1]:
                        index_stack.pop()
                        current_pattern.append(user)
                        find_patterns(user, index_stack, current_pattern, matched_patterns, seen)
        return matched_patterns
    matched_patterns: List[List[Node]] = []
    seen: Set[Tuple[Node, Tuple[int, ...]]] = set()
    for node in graph.nodes:
        find_patterns(node, [], [], matched_patterns, seen)
    for pattern in matched_patterns:
        first_tuple = pattern[0]
        last_getitem = pattern[-1]
        assert first_tuple.op == 'call_function' and first_tuple.target == tuple
        assert last_getitem.op == 'call_function' and last_getitem.target == operator.getitem
        last_getitem_index = last_getitem.args[1]
        new_input = first_tuple.args[0][last_getitem_index]
        for user in list(last_getitem.users.keys()):
            user.replace_input_with(last_getitem, new_input)