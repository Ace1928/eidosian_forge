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
@dataclass
class EmptyLineTracker:
    """Provides a stateful method that returns the number of potential extra
    empty lines needed before and after the currently processed line.

    Note: this tracker works on lines that haven't been split yet.  It assumes
    the prefix of the first leaf consists of optional newlines.  Those newlines
    are consumed by `maybe_empty_lines()` and included in the computation.
    """
    mode: Mode
    previous_line: Optional[Line] = None
    previous_block: Optional[LinesBlock] = None
    previous_defs: List[Line] = field(default_factory=list)
    semantic_leading_comment: Optional[LinesBlock] = None

    def maybe_empty_lines(self, current_line: Line) -> LinesBlock:
        """Return the number of extra empty lines before and after the `current_line`.

        This is for separating `def`, `async def` and `class` with extra empty
        lines (two on module-level).
        """
        form_feed = current_line.depth == 0 and bool(current_line.leaves) and ('\x0c\n' in current_line.leaves[0].prefix)
        before, after = self._maybe_empty_lines(current_line)
        previous_after = self.previous_block.after if self.previous_block else 0
        before = max(0, before - previous_after)
        if self.previous_block and self.previous_block.previous_block is None and (len(self.previous_block.original_line.leaves) == 1) and self.previous_block.original_line.is_docstring and (not (current_line.is_class or current_line.is_def)):
            before = 1
        block = LinesBlock(mode=self.mode, previous_block=self.previous_block, original_line=current_line, before=before, after=after, form_feed=form_feed)
        if current_line.is_comment:
            if self.previous_line is None or (not self.previous_line.is_decorator and (not self.previous_line.is_comment or before) and (self.semantic_leading_comment is None or before)):
                self.semantic_leading_comment = block
        elif not current_line.is_decorator or before:
            self.semantic_leading_comment = None
        self.previous_line = current_line
        self.previous_block = block
        return block

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