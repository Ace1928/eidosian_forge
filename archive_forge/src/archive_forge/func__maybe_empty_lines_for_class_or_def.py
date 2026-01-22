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
def _maybe_empty_lines_for_class_or_def(self, current_line: Line, before: int, user_had_newline: bool) -> Tuple[int, int]:
    assert self.previous_line is not None
    if self.previous_line.is_decorator:
        if self.mode.is_pyi and current_line.is_stub_class:
            return (0, 1)
        return (0, 0)
    if self.previous_line.depth < current_line.depth and (self.previous_line.is_class or self.previous_line.is_def):
        if self.mode.is_pyi:
            return (0, 0)
        return (1 if user_had_newline else 0, 0)
    comment_to_add_newlines: Optional[LinesBlock] = None
    if self.previous_line.is_comment and self.previous_line.depth == current_line.depth and (before == 0):
        slc = self.semantic_leading_comment
        if slc is not None and slc.previous_block is not None and (not slc.previous_block.original_line.is_class) and (not slc.previous_block.original_line.opens_block) and (slc.before <= 1):
            comment_to_add_newlines = slc
        else:
            return (0, 0)
    if self.mode.is_pyi:
        if current_line.is_class or self.previous_line.is_class:
            if self.previous_line.depth < current_line.depth:
                newlines = 0
            elif self.previous_line.depth > current_line.depth:
                newlines = 1
            elif current_line.is_stub_class and self.previous_line.is_stub_class:
                newlines = 0
            else:
                newlines = 1
        elif self.previous_line.depth > current_line.depth:
            newlines = 1
        elif (current_line.is_def or current_line.is_decorator) and (not self.previous_line.is_def):
            if current_line.depth:
                newlines = min(1, before)
            else:
                newlines = 1
        else:
            newlines = 0
    else:
        newlines = 1 if current_line.depth else 2
        if self.previous_line.is_stub_def and (not user_had_newline):
            newlines = 0
    if comment_to_add_newlines is not None:
        previous_block = comment_to_add_newlines.previous_block
        if previous_block is not None:
            comment_to_add_newlines.before = max(comment_to_add_newlines.before, newlines) - previous_block.after
            newlines = 0
    return (newlines, 0)