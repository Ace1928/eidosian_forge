import re
import sys
from dataclasses import replace
from enum import Enum, auto
from functools import partial, wraps
from typing import Collection, Iterator, List, Optional, Set, Union, cast
from black.brackets import (
from black.comments import FMT_OFF, generate_comments, list_comments
from black.lines import (
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.numerics import normalize_numeric_literal
from black.strings import (
from black.trans import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _maybe_split_omitting_optional_parens(rhs: RHSResult, line: Line, mode: Mode, features: Collection[Feature]=(), omit: Collection[LeafID]=()) -> Iterator[Line]:
    if Feature.FORCE_OPTIONAL_PARENTHESES not in features and rhs.opening_bracket.type == token.LPAR and (not rhs.opening_bracket.value) and (rhs.closing_bracket.type == token.RPAR) and (not rhs.closing_bracket.value) and (not line.is_import) and can_omit_invisible_parens(rhs, mode.line_length):
        omit = {id(rhs.closing_bracket), *omit}
        try:
            rhs_oop = _first_right_hand_split(line, omit=omit)
            is_split_right_after_equal = len(rhs.head.leaves) >= 2 and rhs.head.leaves[-2].type == token.EQUAL
            rhs_head_contains_brackets = any((leaf.type in BRACKETS for leaf in rhs.head.leaves[:-1]))
            rhs_head_short_enough = is_line_short_enough(rhs.head, mode=replace(mode, line_length=mode.line_length - 1))
            rhs_head_explode_blocked_by_magic_trailing_comma = rhs.head.magic_trailing_comma is None
            if not (is_split_right_after_equal and rhs_head_contains_brackets and rhs_head_short_enough and rhs_head_explode_blocked_by_magic_trailing_comma) or _prefer_split_rhs_oop_over_rhs(rhs_oop, rhs, mode):
                yield from _maybe_split_omitting_optional_parens(rhs_oop, line, mode, features=features, omit=omit)
                return
        except CannotSplit as e:
            if line.is_chained_assignment:
                pass
            elif not can_be_split(rhs.body) and (not is_line_short_enough(rhs.body, mode=mode)):
                raise CannotSplit("Splitting failed, body is still too long and can't be split.") from e
            elif rhs.head.contains_multiline_strings() or rhs.tail.contains_multiline_strings():
                raise CannotSplit('The current optional pair of parentheses is bound to fail to satisfy the splitting algorithm because the head or the tail contains multiline strings which by definition never fit one line.') from e
    ensure_visible(rhs.opening_bracket)
    ensure_visible(rhs.closing_bracket)
    for result in (rhs.head, rhs.body, rhs.tail):
        if result:
            yield result