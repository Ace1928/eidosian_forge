from __future__ import annotations
import re
from markdown.treeprocessors import Treeprocessor, isString
from markdown.extensions import Extension
from typing import TYPE_CHECKING
class LegacyAttrs(Treeprocessor):

    def run(self, doc: etree.Element) -> None:
        """Find and set values of attributes ({@key=value}). """
        for el in doc.iter():
            alt = el.get('alt', None)
            if alt is not None:
                el.set('alt', self.handleAttributes(el, alt))
            if el.text and isString(el.text):
                el.text = self.handleAttributes(el, el.text)
            if el.tail and isString(el.tail):
                el.tail = self.handleAttributes(el, el.tail)

    def handleAttributes(self, el: etree.Element, txt: str) -> str:
        """ Set attributes and return text without definitions. """

        def attributeCallback(match: re.Match[str]):
            el.set(match.group(1), match.group(2).replace('\n', ' '))
        return ATTR_RE.sub(attributeCallback, txt)