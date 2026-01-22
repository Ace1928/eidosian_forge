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
def _update_vertex_lists(self, update_view_translation=True) -> None:
    style_invalid_start, style_invalid_end = self.invalid_style.validate()
    self.invalid_vertex_lines.invalidate(self.get_line_from_position(style_invalid_start), self.get_line_from_position(style_invalid_end) + 1)
    invalid_start, invalid_end = self.invalid_vertex_lines.validate()
    if invalid_end - invalid_start <= 0:
        return
    colors_iter = self.document.get_style_runs('color')
    background_iter = self.document.get_style_runs('background_color')
    if self._selection_end - self._selection_start > 0:
        colors_iter = runlist.OverriddenRunIterator(colors_iter, self._selection_start, self._selection_end, self._selection_color)
        background_iter = runlist.OverriddenRunIterator(background_iter, self._selection_start, self._selection_end, self._selection_background_color)
    context = _IncrementalLayoutContext(self, self._document, colors_iter, background_iter)
    lines = self.lines[invalid_start:invalid_end]
    self._line_count = len(self.lines)
    self._ascent = lines[0].ascent
    self._descent = lines[0].descent
    self._anchor_left = self._get_left_anchor()
    self._anchor_bottom = self._get_bottom_anchor()
    top_anchor = self._get_top_anchor()
    for line in lines:
        line.delete(self)
        context.line = line
        y = line.y
        if y + line.descent > self._translate_y:
            continue
        elif y + line.ascent < self._translate_y - self.height:
            break
        self._create_vertex_lists(line.x, y, self._anchor_left, top_anchor, line.start, line.boxes, context)
    if update_view_translation:
        self._update_view_translation()