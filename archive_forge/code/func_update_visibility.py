from __future__ import annotations
import re
from typing import List, TYPE_CHECKING, Optional, Any
import pyglet
import pyglet.text.layout
from pyglet.gl import GL_TEXTURE0, glActiveTexture, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
def update_visibility(self, visible: bool) -> None:
    visible_tuple = (visible,)
    for _vertex_list in self.vertex_lists.values():
        _vertex_list.visible[:] = visible_tuple * _vertex_list.count