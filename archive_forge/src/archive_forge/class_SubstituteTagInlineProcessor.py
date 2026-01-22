from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class SubstituteTagInlineProcessor(SimpleTagInlineProcessor):
    """ Return an element of type `tag` with no children. """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        """ Return empty [`Element`][xml.etree.ElementTree.Element] of type `tag`. """
        return (etree.Element(self.tag), m.start(0), m.end(0))