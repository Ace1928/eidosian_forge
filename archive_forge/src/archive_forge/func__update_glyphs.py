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
def _update_glyphs(self) -> None:
    invalid_start, invalid_end = self.invalid_glyphs.validate()
    if invalid_end - invalid_start <= 0:
        return
    text = self.document.text
    while invalid_start > 0:
        if grapheme_break(text[invalid_start - 1], text[invalid_start]):
            break
        invalid_start -= 1
    len_text = len(text)
    while invalid_end < len_text:
        if grapheme_break(text[invalid_end - 1], text[invalid_end]):
            break
        invalid_end += 1
    runs = runlist.ZipRunIterator((self._document.get_font_runs(dpi=self._dpi), self._document.get_element_runs()))
    for start, end, (font, element) in runs.ranges(invalid_start, invalid_end):
        if element:
            self.glyphs[start] = _InlineElementBox(element)
        else:
            text = self.document.text[start:end]
            self.glyphs[start:end] = font.get_glyphs(text)
    self._get_owner_runs(self.owner_runs, self.glyphs, invalid_start, invalid_end)
    self.invalid_flow.invalidate(invalid_start, invalid_end)