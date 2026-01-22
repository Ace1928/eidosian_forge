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