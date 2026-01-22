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
def _get_bottom_anchor(self) -> float:
    """Returns the anchor for the Y axis from the bottom."""
    height = self._height
    if self._content_valign == 'top':
        offset = min(0, self._height)
    elif self._content_valign == 'bottom':
        offset = 0
    elif self._content_valign == 'center':
        offset = min(0, self._height) // 2
    else:
        assert False, '`content_valign` must be either "top", "bottom", or "center".'
    if self._anchor_y == 'top':
        return -height + offset
    elif self._anchor_y == 'baseline':
        return -height + self._ascent
    elif self._anchor_y == 'bottom':
        return 0
    elif self._anchor_y == 'center':
        if self._line_count == 1 and self._height is None:
            return self._ascent // 2 - self._descent // 4 - height
        else:
            return offset - height // 2
    else:
        assert False, '`anchor_y` must be either "top", "bottom", "center", or "baseline".'