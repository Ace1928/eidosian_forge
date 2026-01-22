from __future__ import annotations
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from . import inlinepatterns
class UnescapeTreeprocessor(Treeprocessor):
    """ Restore escaped chars """
    RE = re.compile('{}(\\d+){}'.format(util.STX, util.ETX))

    def _unescape(self, m: re.Match[str]) -> str:
        return chr(int(m.group(1)))

    def unescape(self, text: str) -> str:
        return self.RE.sub(self._unescape, text)

    def run(self, root: etree.Element) -> None:
        """ Loop over all elements and unescape all text. """
        for elem in root.iter():
            if elem.text and (not elem.tag == 'code'):
                elem.text = self.unescape(elem.text)
            if elem.tail:
                elem.tail = self.unescape(elem.tail)
            for key, value in elem.items():
                elem.set(key, self.unescape(value))