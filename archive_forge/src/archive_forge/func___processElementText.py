from __future__ import annotations
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from . import inlinepatterns
def __processElementText(self, node: etree.Element, subnode: etree.Element, isText: bool=True) -> None:
    """
        Process placeholders in `Element.text` or `Element.tail`
        of Elements popped from `self.stashed_nodes`.

        Arguments:
            node: Parent node.
            subnode: Processing node.
            isText: Boolean variable, True - it's text, False - it's a tail.

        """
    if isText:
        text = subnode.text
        subnode.text = None
    else:
        text = subnode.tail
        subnode.tail = None
    childResult = self.__processPlaceholders(text, subnode, isText)
    if not isText and node is not subnode:
        pos = list(node).index(subnode) + 1
    else:
        pos = 0
    childResult.reverse()
    for newChild in childResult:
        node.insert(pos, newChild[0])