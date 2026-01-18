from __future__ import annotations
import re
from typing import List, TYPE_CHECKING, Optional, Any
import pyglet
import pyglet.text.layout
from pyglet.gl import GL_TEXTURE0, glActiveTexture, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
def update_view_translation(self, translate_x: float, translate_y: float) -> None:
    view_translation = (-translate_x, -translate_y, 0)
    for _vertex_list in self.vertex_lists.values():
        _vertex_list.view_translation[:] = view_translation * _vertex_list.count