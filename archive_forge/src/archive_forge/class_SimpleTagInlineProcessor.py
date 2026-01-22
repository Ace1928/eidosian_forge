from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class SimpleTagInlineProcessor(InlineProcessor):
    """
    Return element of type `tag` with a text attribute of `group(2)`
    of a Pattern.

    """

    def __init__(self, pattern: str, tag: str):
        """
        Create an instant of an simple tag processor.

        Arguments:
            pattern: A regular expression that matches a pattern.
            tag: Tag of element.

        """
        InlineProcessor.__init__(self, pattern)
        self.tag = tag
        ' The tag of the rendered element. '

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        """
        Return [`Element`][xml.etree.ElementTree.Element] of type `tag` with the string in `group(2)` of a
        matching pattern as the Element's text.
        """
        el = etree.Element(self.tag)
        el.text = m.group(2)
        return (el, m.start(0), m.end(0))