from __future__ import annotations
import sys
from typing import List, Optional, Type, Any, Tuple, TYPE_CHECKING
from pyglet.customtypes import AnchorX, AnchorY
from pyglet.event import EventDispatcher
from pyglet.font.base import grapheme_break
from pyglet.text import runlist
from pyglet.text.document import AbstractDocument
from pyglet.text.layout.base import _is_pyglet_doc_run, _Line, _LayoutContext, _InlineElementBox, _InvalidRange, \
from pyglet.text.layout.scrolling import ScrollableTextLayoutGroup, ScrollableTextDecorationGroup
def _update_flow_glyphs(self) -> None:
    invalid_start, invalid_end = self.invalid_flow.validate()
    if invalid_end - invalid_start <= 0:
        return
    line_index = 0
    for i, line in enumerate(self.lines):
        if line.start >= invalid_start:
            break
        line_index = i
    line_index = max(0, line_index - 1)
    try:
        line = self.lines[line_index]
        invalid_start = min(invalid_start, line.start)
        line.delete(self)
        self.lines[line_index] = _Line(invalid_start)
        self.invalid_lines.invalidate(line_index, line_index + 1)
    except IndexError:
        line_index = 0
        invalid_start = 0
        line = _Line(0)
        self.lines.append(line)
        self.invalid_lines.insert(0, 1)
    content_width_invalid = False
    next_start = invalid_start
    for line in self._flow_glyphs(self.glyphs, self.owner_runs, invalid_start, len(self._document.text)):
        try:
            old_line = self.lines[line_index]
            old_line.delete(self)
            old_line_width = old_line.width + old_line.margin_left
            new_line_width = line.width + line.margin_left
            if old_line_width == self._content_width and new_line_width < old_line_width:
                content_width_invalid = True
            self.lines[line_index] = line
            self.invalid_lines.invalidate(line_index, line_index + 1)
        except IndexError:
            self.lines.append(line)
            self.invalid_lines.insert(line_index, 1)
        next_start = line.start + line.length
        line_index += 1
        try:
            next_line = self.lines[line_index]
            if next_start == next_line.start and next_start > invalid_end:
                break
        except IndexError:
            pass
    if next_start == len(self._document.text) and line_index > 0:
        for line in self.lines[line_index:]:
            old_line_width = old_line.width + old_line.margin_left
            if old_line_width == self._content_width:
                content_width_invalid = True
            line.delete(self)
        del self.lines[line_index:]
    if content_width_invalid or len(self.lines) == 1:
        content_width = 0
        for line in self.lines:
            content_width = max(line.width + line.margin_left, content_width)
        self._content_width = content_width