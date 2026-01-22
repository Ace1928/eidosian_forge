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
class BaseStringSplitter(StringTransformer):
    """
    Abstract class for StringTransformers which transform a Line's strings by splitting
    them or placing them on their own lines where necessary to avoid going over
    the configured line length.

    Requirements:
        * The target string value is responsible for the line going over the
          line length limit. It follows that after all of black's other line
          split methods have been exhausted, this line (or one of the resulting
          lines after all line splits are performed) would still be over the
          line_length limit unless we split this string.
          AND

        * The target string is NOT a "pointless" string (i.e. a string that has
          no parent or siblings).
          AND

        * The target string is not followed by an inline comment that appears
          to be a pragma.
          AND

        * The target string is not a multiline (i.e. triple-quote) string.
    """
    STRING_OPERATORS: Final = [token.EQEQUAL, token.GREATER, token.GREATEREQUAL, token.LESS, token.LESSEQUAL, token.NOTEQUAL, token.PERCENT, token.PLUS, token.STAR]

    @abstractmethod
    def do_splitter_match(self, line: Line) -> TMatchResult:
        """
        BaseStringSplitter asks its clients to override this method instead of
        `StringTransformer.do_match(...)`.

        Follows the same protocol as `StringTransformer.do_match(...)`.

        Refer to `help(StringTransformer.do_match)` for more information.
        """

    def do_match(self, line: Line) -> TMatchResult:
        match_result = self.do_splitter_match(line)
        if isinstance(match_result, Err):
            return match_result
        string_indices = match_result.ok()
        assert len(string_indices) == 1, f'{self.__class__.__name__} should only find one match at a time, found {len(string_indices)}'
        string_idx = string_indices[0]
        vresult = self._validate(line, string_idx)
        if isinstance(vresult, Err):
            return vresult
        return match_result

    def _validate(self, line: Line, string_idx: int) -> TResult[None]:
        """
        Checks that @line meets all of the requirements listed in this classes'
        docstring. Refer to `help(BaseStringSplitter)` for a detailed
        description of those requirements.

        Returns:
            * Ok(None), if ALL of the requirements are met.
              OR
            * Err(CannotTransform), if ANY of the requirements are NOT met.
        """
        LL = line.leaves
        string_leaf = LL[string_idx]
        max_string_length = self._get_max_string_length(line, string_idx)
        if len(string_leaf.value) <= max_string_length:
            return TErr('The string itself is not what is causing this line to be too long.')
        if not string_leaf.parent or [L.type for L in string_leaf.parent.children] == [token.STRING, token.NEWLINE]:
            return TErr(f'This string ({string_leaf.value}) appears to be pointless (i.e. has no parent).')
        if id(line.leaves[string_idx]) in line.comments and contains_pragma_comment(line.comments[id(line.leaves[string_idx])]):
            return TErr("Line appears to end with an inline pragma comment. Splitting the line could modify the pragma's behavior.")
        if has_triple_quotes(string_leaf.value):
            return TErr('We cannot split multiline strings.')
        return Ok(None)

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

    @staticmethod
    def _prefer_paren_wrap_match(LL: List[Leaf]) -> Optional[int]:
        """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the "prefer paren wrap" statement
            requirements listed in the 'Requirements' section of the StringParenWrapper
            class's docstring.
                OR
            None, otherwise.
        """
        if LL[0].type != token.STRING:
            return None
        matching_nodes = [syms.listmaker, syms.dictsetmaker, syms.testlist_gexp]
        if parent_type(LL[0]) in matching_nodes or parent_type(LL[0].parent) in matching_nodes:
            prev_sibling = LL[0].prev_sibling
            next_sibling = LL[0].next_sibling
            if not prev_sibling and (not next_sibling) and (parent_type(LL[0]) == syms.atom):
                parent = LL[0].parent
                assert parent is not None
                prev_sibling = parent.prev_sibling
                next_sibling = parent.next_sibling
            if (not prev_sibling or prev_sibling.type == token.COMMA) and (not next_sibling or next_sibling.type == token.COMMA):
                return 0
        return None