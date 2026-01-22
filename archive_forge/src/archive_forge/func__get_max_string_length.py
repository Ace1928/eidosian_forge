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
def _get_max_string_length(self, line: Line, string_idx: int) -> int:
    """
        Calculates the max string length used when attempting to determine
        whether or not the target string is responsible for causing the line to
        go over the line length limit.

        WARNING: This method is tightly coupled to both StringSplitter and
        (especially) StringParenWrapper. There is probably a better way to
        accomplish what is being done here.

        Returns:
            max_string_length: such that `line.leaves[string_idx].value >
            max_string_length` implies that the target string IS responsible
            for causing this line to exceed the line length limit.
        """
    LL = line.leaves
    is_valid_index = is_valid_index_factory(LL)
    offset = line.depth * 4
    if is_valid_index(string_idx - 1):
        p_idx = string_idx - 1
        if LL[string_idx - 1].type == token.LPAR and LL[string_idx - 1].value == '' and (string_idx >= 2):
            p_idx -= 1
        P = LL[p_idx]
        if P.type in self.STRING_OPERATORS:
            offset += len(str(P)) + 1
        if P.type == token.COMMA:
            offset += 3
        if P.type in [token.COLON, token.EQUAL, token.PLUSEQUAL, token.NAME]:
            offset += 1
            for leaf in reversed(LL[:p_idx + 1]):
                offset += len(str(leaf))
                if leaf.type in CLOSING_BRACKETS:
                    break
    if is_valid_index(string_idx + 1):
        N = LL[string_idx + 1]
        if N.type == token.RPAR and N.value == '' and (len(LL) > string_idx + 2):
            N = LL[string_idx + 2]
        if N.type == token.COMMA:
            offset += 1
        if is_valid_index(string_idx + 2):
            NN = LL[string_idx + 2]
            if N.type == token.DOT and NN.type == token.NAME:
                offset += 1
                if is_valid_index(string_idx + 3) and LL[string_idx + 3].type == token.LPAR:
                    offset += 1
                offset += len(NN.value)
    has_comments = False
    for comment_leaf in line.comments_after(LL[string_idx]):
        if not has_comments:
            has_comments = True
            offset += 2
        offset += len(comment_leaf.value)
    max_string_length = count_chars_in_width(str(line), self.line_length - offset)
    return max_string_length