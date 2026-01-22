from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
class ReferenceProcessor(BlockProcessor):
    """ Process link references. """
    RE = re.compile('^[ ]{0,3}\\[([^\\[\\]]*)\\]:[ ]*\\n?[ ]*([^\\s]+)[ ]*(?:\\n[ ]*)?((["\\\'])(.*)\\4[ ]*|\\((.*)\\)[ ]*)?$', re.MULTILINE)

    def test(self, parent: etree.Element, block: str) -> bool:
        return True

    def run(self, parent: etree.Element, blocks: list[str]) -> bool:
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            id = m.group(1).strip().lower()
            link = m.group(2).lstrip('<').rstrip('>')
            title = m.group(5) or m.group(6)
            self.parser.md.references[id] = (link, title)
            if block[m.end():].strip():
                blocks.insert(0, block[m.end():].lstrip('\n'))
            if block[:m.start()].strip():
                blocks.insert(0, block[:m.start()].rstrip('\n'))
            return True
        blocks.insert(0, block)
        return False