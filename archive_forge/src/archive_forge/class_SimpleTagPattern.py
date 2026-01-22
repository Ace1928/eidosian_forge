from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class SimpleTagPattern(Pattern):
    """
    Return element of type `tag` with a text attribute of `group(3)`
    of a Pattern.

    """

    def __init__(self, pattern: str, tag: str):
        """
        Create an instant of an simple tag pattern.

        Arguments:
            pattern: A regular expression that matches a pattern.
            tag: Tag of element.

        """
        Pattern.__init__(self, pattern)
        self.tag = tag
        ' The tag of the rendered element. '

    def handleMatch(self, m: re.Match[str]) -> etree.Element:
        """
        Return [`Element`][xml.etree.ElementTree.Element] of type `tag` with the string in `group(3)` of a
        matching pattern as the Element's text.
        """
        el = etree.Element(self.tag)
        el.text = m.group(3)
        return el