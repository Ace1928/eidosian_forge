from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Pattern, Union, Optional, List, Any, Tuple, Callable, Iterator, Type, Dict, \
import pyglet
from pyglet import graphics
from pyglet.customtypes import AnchorX, AnchorY, ContentVAlign, HorizontalAlign
from pyglet.font.base import Font, Glyph
from pyglet.gl import GL_TRIANGLES, GL_LINES, glActiveTexture, GL_TEXTURE0, glBindTexture, glEnable, GL_BLEND, \
from pyglet.image import Texture
from pyglet.text import runlist
from pyglet.text.runlist import RunIterator, AbstractRunIterator
def _get_top_anchor(self) -> float:
    """Returns the anchor for the Y axis from the top."""
    if self._height is None:
        height = self._content_height
        offset = 0
    else:
        height = self._height
        if self._content_valign == 'top':
            offset = 0
        elif self._content_valign == 'bottom':
            offset = max(0, self._height - self._content_height)
        elif self._content_valign == 'center':
            offset = max(0, self._height - self._content_height) // 2
        else:
            assert False, '`content_valign` must be either "top", "bottom", or "center".'
    if self._anchor_y == 'top':
        return -offset
    elif self._anchor_y == 'baseline':
        return self._ascent - offset
    elif self._anchor_y == 'bottom':
        return height - offset
    elif self._anchor_y == 'center':
        if self._line_count == 1 and self._height is None:
            return self._ascent // 2 - self._descent // 4
        else:
            return height // 2 - offset
    else:
        assert False, '`anchor_y` must be either "top", "bottom", "center", or "baseline".'