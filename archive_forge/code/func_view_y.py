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
@view_y.setter
def view_y(self, view_y: int) -> None:
    translation = min(0, max(self.height - self._content_height, view_y))
    if translation != self._translate_y:
        self._translate_y = translation
        self._update_visible_lines()
        self._update_vertex_lists(update_view_translation=False)
        self._update_view_translation()