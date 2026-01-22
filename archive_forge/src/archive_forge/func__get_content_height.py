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
def _get_content_height(self) -> int:
    """Returns the height of the layout content factoring in the vertical alignment."""
    if self._height is None:
        height = self.content_height
        offset = 0
    else:
        height = self._height
        if self._content_valign == 'top':
            offset = 0
        elif self._content_valign == 'bottom':
            offset = max(0, self._height - self.content_height)
        elif self._content_valign == 'center':
            offset = max(0, self._height - self.content_height) // 2
        else:
            assert False, '`content_valign` must be either "top", "bottom", or "center".'
    return height - offset