import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
class StringMerger(StringTransformer, CustomSplitMapMixin):
    """StringTransformer that merges strings together.

    Requirements:
        (A) The line contains adjacent strings such that ALL of the validation checks
        listed in StringMerger._validate_msg(...)'s docstring pass.
        OR
        (B) The line contains a string which uses line continuation backslashes.

    Transformations:
        Depending on which of the two requirements above where met, either:

        (A) The string group associated with the target string is merged.
        OR
        (B) All line-continuation backslashes are removed from the target string.

    Collaborations:
        StringMerger provides custom split information to StringSplitter.
    """

    def do_match(self, line: Line) -> TMatchResult:
        LL = line.leaves
        is_valid_index = is_valid_index_factory(LL)
        string_indices = []
        idx = 0
        while is_valid_index(idx):
            leaf = LL[idx]
            if leaf.type == token.STRING and is_valid_index(idx + 1) and (LL[idx + 1].type == token.STRING):
                contains_comment = False
                i = idx
                while is_valid_index(i):
                    if LL[i].type != token.STRING:
                        break
                    if line.comments_after(LL[i]):
                        contains_comment = True
                        break
                    i += 1
                if not is_part_of_annotation(leaf) and (not contains_comment):
                    string_indices.append(idx)
                idx += 2
                while is_valid_index(idx) and LL[idx].type == token.STRING:
                    idx += 1
            elif leaf.type == token.STRING and '\\\n' in leaf.value:
                string_indices.append(idx)
                idx += 1
                while is_valid_index(idx) and LL[idx].type == token.STRING:
                    idx += 1
            else:
                idx += 1
        if string_indices:
            return Ok(string_indices)
        else:
            return TErr('This line has no strings that need merging.')

    def do_transform(self, line: Line, string_indices: List[int]) -> Iterator[TResult[Line]]:
        new_line = line
        rblc_result = self._remove_backslash_line_continuation_chars(new_line, string_indices)
        if isinstance(rblc_result, Ok):
            new_line = rblc_result.ok()
        msg_result = self._merge_string_group(new_line, string_indices)
        if isinstance(msg_result, Ok):
            new_line = msg_result.ok()
        if isinstance(rblc_result, Err) and isinstance(msg_result, Err):
            msg_cant_transform = msg_result.err()
            rblc_cant_transform = rblc_result.err()
            cant_transform = CannotTransform('StringMerger failed to merge any strings in this line.')
            msg_cant_transform.__cause__ = rblc_cant_transform
            cant_transform.__cause__ = msg_cant_transform
            yield Err(cant_transform)
        else:
            yield Ok(new_line)

    @staticmethod
    def _remove_backslash_line_continuation_chars(line: Line, string_indices: List[int]) -> TResult[Line]:
        """
        Merge strings that were split across multiple lines using
        line-continuation backslashes.

        Returns:
            Ok(new_line), if @line contains backslash line-continuation
            characters.
                OR
            Err(CannotTransform), otherwise.
        """
        LL = line.leaves
        indices_to_transform = []
        for string_idx in string_indices:
            string_leaf = LL[string_idx]
            if string_leaf.type == token.STRING and '\\\n' in string_leaf.value and (not has_triple_quotes(string_leaf.value)):
                indices_to_transform.append(string_idx)
        if not indices_to_transform:
            return TErr('Found no string leaves that contain backslash line continuation characters.')
        new_line = line.clone()
        new_line.comments = line.comments.copy()
        append_leaves(new_line, line, LL)
        for string_idx in indices_to_transform:
            new_string_leaf = new_line.leaves[string_idx]
            new_string_leaf.value = new_string_leaf.value.replace('\\\n', '')
        return Ok(new_line)

    def _merge_string_group(self, line: Line, string_indices: List[int]) -> TResult[Line]:
        """
        Merges string groups (i.e. set of adjacent strings).

        Each index from `string_indices` designates one string group's first
        leaf in `line.leaves`.

        Returns:
            Ok(new_line), if ALL of the validation checks found in
            _validate_msg(...) pass.
                OR
            Err(CannotTransform), otherwise.
        """
        LL = line.leaves
        is_valid_index = is_valid_index_factory(LL)
        merged_string_idx_dict: Dict[int, Tuple[int, Leaf]] = {}
        for string_idx in string_indices:
            vresult = self._validate_msg(line, string_idx)
            if isinstance(vresult, Err):
                continue
            merged_string_idx_dict[string_idx] = self._merge_one_string_group(LL, string_idx, is_valid_index)
        if not merged_string_idx_dict:
            return TErr('No string group is merged')
        new_line = line.clone()
        previous_merged_string_idx = -1
        previous_merged_num_of_strings = -1
        for i, leaf in enumerate(LL):
            if i in merged_string_idx_dict:
                previous_merged_string_idx = i
                previous_merged_num_of_strings, string_leaf = merged_string_idx_dict[i]
                new_line.append(string_leaf)
            if previous_merged_string_idx <= i < previous_merged_string_idx + previous_merged_num_of_strings:
                for comment_leaf in line.comments_after(LL[i]):
                    new_line.append(comment_leaf, preformatted=True)
                continue
            append_leaves(new_line, line, [leaf])
        return Ok(new_line)

    def _merge_one_string_group(self, LL: List[Leaf], string_idx: int, is_valid_index: Callable[[int], bool]) -> Tuple[int, Leaf]:
        """
        Merges one string group where the first string in the group is
        `LL[string_idx]`.

        Returns:
            A tuple of `(num_of_strings, leaf)` where `num_of_strings` is the
            number of strings merged and `leaf` is the newly merged string
            to be replaced in the new line.
        """
        atom_node = LL[string_idx].parent
        BREAK_MARK = '@@@@@ BLACK BREAKPOINT MARKER @@@@@'
        QUOTE = LL[string_idx].value[-1]

        def make_naked(string: str, string_prefix: str) -> str:
            """Strip @string (i.e. make it a "naked" string)

            Pre-conditions:
                * assert_is_leaf_string(@string)

            Returns:
                A string that is identical to @string except that
                @string_prefix has been stripped, the surrounding QUOTE
                characters have been removed, and any remaining QUOTE
                characters have been escaped.
            """
            assert_is_leaf_string(string)
            if 'f' in string_prefix:
                f_expressions = (string[span[0] + 1:span[1] - 1] for span in iter_fexpr_spans(string))
                debug_expressions_contain_visible_quotes = any((re.search('.*[\\\'\\"].*(?<![!:=])={1}(?!=)(?![^\\s:])', expression) for expression in f_expressions))
                if not debug_expressions_contain_visible_quotes:
                    string = _toggle_fexpr_quotes(string, QUOTE)
            RE_EVEN_BACKSLASHES = '(?:(?<!\\\\)(?:\\\\\\\\)*)'
            naked_string = string[len(string_prefix) + 1:-1]
            naked_string = re.sub('(' + RE_EVEN_BACKSLASHES + ')' + QUOTE, '\\1\\\\' + QUOTE, naked_string)
            return naked_string
        custom_splits = []
        prefix_tracker = []
        next_str_idx = string_idx
        prefix = ''
        while not prefix and is_valid_index(next_str_idx) and (LL[next_str_idx].type == token.STRING):
            prefix = get_string_prefix(LL[next_str_idx].value).lower()
            next_str_idx += 1
        S = ''
        NS = ''
        num_of_strings = 0
        next_str_idx = string_idx
        while is_valid_index(next_str_idx) and LL[next_str_idx].type == token.STRING:
            num_of_strings += 1
            SS = LL[next_str_idx].value
            next_prefix = get_string_prefix(SS).lower()
            if 'f' in prefix and 'f' not in next_prefix:
                SS = re.sub('(\\{|\\})', '\\1\\1', SS)
            NSS = make_naked(SS, next_prefix)
            has_prefix = bool(next_prefix)
            prefix_tracker.append(has_prefix)
            S = prefix + QUOTE + NS + NSS + BREAK_MARK + QUOTE
            NS = make_naked(S, prefix)
            next_str_idx += 1
        non_string_idx = next_str_idx
        S_leaf = Leaf(token.STRING, S)
        if self.normalize_strings:
            S_leaf.value = normalize_string_quotes(S_leaf.value)
        temp_string = S_leaf.value[len(prefix) + 1:-1]
        for has_prefix in prefix_tracker:
            mark_idx = temp_string.find(BREAK_MARK)
            assert mark_idx >= 0, 'Logic error while filling the custom string breakpoint cache.'
            temp_string = temp_string[mark_idx + len(BREAK_MARK):]
            breakpoint_idx = mark_idx + (len(prefix) if has_prefix else 0) + 1
            custom_splits.append(CustomSplit(has_prefix, breakpoint_idx))
        string_leaf = Leaf(token.STRING, S_leaf.value.replace(BREAK_MARK, ''))
        if atom_node is not None:
            if non_string_idx - string_idx < len(atom_node.children):
                first_child_idx = LL[string_idx].remove()
                for idx in range(string_idx + 1, non_string_idx):
                    LL[idx].remove()
                if first_child_idx is not None:
                    atom_node.insert_child(first_child_idx, string_leaf)
            else:
                replace_child(atom_node, string_leaf)
        self.add_custom_splits(string_leaf.value, custom_splits)
        return (num_of_strings, string_leaf)

    @staticmethod
    def _validate_msg(line: Line, string_idx: int) -> TResult[None]:
        """Validate (M)erge (S)tring (G)roup

        Transform-time string validation logic for _merge_string_group(...).

        Returns:
            * Ok(None), if ALL validation checks (listed below) pass.
                OR
            * Err(CannotTransform), if any of the following are true:
                - The target string group does not contain ANY stand-alone comments.
                - The target string is not in a string group (i.e. it has no
                  adjacent strings).
                - The string group has more than one inline comment.
                - The string group has an inline comment that appears to be a pragma.
                - The set of all string prefixes in the string group is of
                  length greater than one and is not equal to {"", "f"}.
                - The string group consists of raw strings.
                - The string group is stringified type annotations. We don't want to
                  process stringified type annotations since pyright doesn't support
                  them spanning multiple string values. (NOTE: mypy, pytype, pyre do
                  support them, so we can change if pyright also gains support in the
                  future. See https://github.com/microsoft/pyright/issues/4359.)
        """
        for inc in [1, -1]:
            i = string_idx
            found_sa_comment = False
            is_valid_index = is_valid_index_factory(line.leaves)
            while is_valid_index(i) and line.leaves[i].type in [token.STRING, STANDALONE_COMMENT]:
                if line.leaves[i].type == STANDALONE_COMMENT:
                    found_sa_comment = True
                elif found_sa_comment:
                    return TErr('StringMerger does NOT merge string groups which contain stand-alone comments.')
                i += inc
        num_of_inline_string_comments = 0
        set_of_prefixes = set()
        num_of_strings = 0
        for leaf in line.leaves[string_idx:]:
            if leaf.type != token.STRING:
                if leaf.type == token.COMMA and id(leaf) in line.comments:
                    num_of_inline_string_comments += 1
                break
            if has_triple_quotes(leaf.value):
                return TErr('StringMerger does NOT merge multiline strings.')
            num_of_strings += 1
            prefix = get_string_prefix(leaf.value).lower()
            if 'r' in prefix:
                return TErr('StringMerger does NOT merge raw strings.')
            set_of_prefixes.add(prefix)
            if id(leaf) in line.comments:
                num_of_inline_string_comments += 1
                if contains_pragma_comment(line.comments[id(leaf)]):
                    return TErr('Cannot merge strings which have pragma comments.')
        if num_of_strings < 2:
            return TErr(f'Not enough strings to merge (num_of_strings={num_of_strings}).')
        if num_of_inline_string_comments > 1:
            return TErr(f'Too many inline string comments ({num_of_inline_string_comments}).')
        if len(set_of_prefixes) > 1 and set_of_prefixes != {'', 'f'}:
            return TErr(f'Too many different prefixes ({set_of_prefixes}).')
        return Ok(None)