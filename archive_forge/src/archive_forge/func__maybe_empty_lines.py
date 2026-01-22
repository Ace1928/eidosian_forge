import itertools
import math
from dataclasses import dataclass, field
from typing import (
from black.brackets import COMMA_PRIORITY, DOT_PRIORITY, BracketTracker
from black.mode import Mode, Preview
from black.nodes import (
from black.strings import str_width
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _maybe_empty_lines(self, current_line: Line) -> Tuple[int, int]:
    max_allowed = 1
    if current_line.depth == 0:
        max_allowed = 1 if self.mode.is_pyi else 2
    if current_line.leaves:
        first_leaf = current_line.leaves[0]
        before = first_leaf.prefix.count('\n')
        before = min(before, max_allowed)
        first_leaf.prefix = ''
    else:
        before = 0
    user_had_newline = bool(before)
    depth = current_line.depth
    previous_def = None
    while self.previous_defs and self.previous_defs[-1].depth >= depth:
        previous_def = self.previous_defs.pop()
    if current_line.is_def or current_line.is_class:
        self.previous_defs.append(current_line)
    if self.previous_line is None:
        return (0, 0)
    if current_line.is_docstring:
        if self.previous_line.is_class:
            return (0, 1)
        if self.previous_line.opens_block and self.previous_line.is_def:
            return (0, 0)
    if previous_def is not None:
        assert self.previous_line is not None
        if self.mode.is_pyi:
            if previous_def.is_class and (not previous_def.is_stub_class):
                before = 1
            elif depth and (not current_line.is_def) and self.previous_line.is_def:
                before = 1 if user_had_newline else 0
            elif depth:
                before = 0
            else:
                before = 1
        elif depth:
            before = 1
        elif not depth and previous_def.depth and (current_line.leaves[-1].type == token.COLON) and (current_line.leaves[0].value not in ('with', 'try', 'for', 'while', 'if', 'match')):
            before = 1
        else:
            before = 2
    if current_line.is_decorator or current_line.is_def or current_line.is_class:
        return self._maybe_empty_lines_for_class_or_def(current_line, before, user_had_newline)
    if self.previous_line.is_import and (not current_line.is_import) and (not current_line.is_fmt_pass_converted(first_leaf_matches=is_import)) and (depth == self.previous_line.depth):
        return (before or 1, 0)
    return (before, 0)