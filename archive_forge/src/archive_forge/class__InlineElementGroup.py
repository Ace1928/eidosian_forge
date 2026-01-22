from __future__ import annotations
import re
from typing import List, TYPE_CHECKING, Optional, Any
import pyglet
import pyglet.text.layout
from pyglet.gl import GL_TEXTURE0, glActiveTexture, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
class _InlineElementGroup(pyglet.graphics.Group):

    def __init__(self, texture, program, order=0, parent=None):
        super().__init__(order, parent)
        self.texture = texture
        self.program = program

    def set_state(self):
        self.program.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(self.texture.target, self.texture.id)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def unset_state(self):
        glDisable(GL_BLEND)
        self.program.stop()

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self._order == other.order and (self.program == other.program) and (self.parent == other.parent) and (self.texture.target == other.texture.target) and (self.texture.id == other.texture.id)

    def __hash__(self):
        return hash((self._order, self.program, self.parent, self.texture.target, self.texture.id))