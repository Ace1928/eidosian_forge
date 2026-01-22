from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
class SetextHeaderProcessor(BlockProcessor):
    """ Process Setext-style Headers. """
    RE = re.compile('^.*?\\n[=-]+[ ]*(\\n|$)', re.MULTILINE)

    def test(self, parent: etree.Element, block: str) -> bool:
        return bool(self.RE.match(block))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        lines = blocks.pop(0).split('\n')
        if lines[1].startswith('='):
            level = 1
        else:
            level = 2
        h = etree.SubElement(parent, 'h%d' % level)
        h.text = lines[0].strip()
        if len(lines) > 2:
            blocks.insert(0, '\n'.join(lines[2:]))