from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def addFormattingElement(self, token):
    self.tree.insertElement(token)
    element = self.tree.openElements[-1]
    matchingElements = []
    for node in self.tree.activeFormattingElements[::-1]:
        if node is Marker:
            break
        elif self.isMatchingFormattingElement(node, element):
            matchingElements.append(node)
    assert len(matchingElements) <= 3
    if len(matchingElements) == 3:
        self.tree.activeFormattingElements.remove(matchingElements[-1])
    self.tree.activeFormattingElements.append(element)