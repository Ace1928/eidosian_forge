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
@view_x.setter
def view_x(self, view_x: int) -> None:
    translation = max(0, min(self._content_width - self._width, view_x))
    if translation != self._translate_x:
        self._translate_x = translation
        self._update_view_translation()