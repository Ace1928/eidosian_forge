from __future__ import annotations
import re
from typing import List, TYPE_CHECKING, Optional, Any
import pyglet
import pyglet.text.layout
from pyglet.gl import GL_TEXTURE0, glActiveTexture, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
class UnorderedListBuilder(ListBuilder):

    def __init__(self, mark):
        """Create an unordered list with constant mark text.

        :Parameters:
            `mark` : str
                Mark to prepend to each list item.

        """
        self.mark = mark

    def get_mark(self, value):
        return self.mark