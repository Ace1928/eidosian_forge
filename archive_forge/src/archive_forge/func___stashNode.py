from __future__ import annotations
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from . import inlinepatterns
def __stashNode(self, node: etree.Element | str, type: str) -> str:
    """ Add node to stash. """
    placeholder, id = self.__makePlaceholder(type)
    self.stashed_nodes[id] = node
    return placeholder