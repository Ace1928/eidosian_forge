from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
class ListIndentProcessor(BlockProcessor):
    """ Process children of list items.

    Example

        * a list item
            process this part

            or this part

    """
    ITEM_TYPES = ['li']
    ' List of tags used for list items. '
    LIST_TYPES = ['ul', 'ol']
    ' Types of lists this processor can operate on. '

    def __init__(self, *args):
        super().__init__(*args)
        self.INDENT_RE = re.compile('^(([ ]{%s})+)' % self.tab_length)

    def test(self, parent: etree.Element, block: str) -> bool:
        return block.startswith(' ' * self.tab_length) and (not self.parser.state.isstate('detabbed')) and (parent.tag in self.ITEM_TYPES or (len(parent) and parent[-1] is not None and (parent[-1].tag in self.LIST_TYPES)))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        level, sibling = self.get_level(parent, block)
        block = self.looseDetab(block, level)
        self.parser.state.set('detabbed')
        if parent.tag in self.ITEM_TYPES:
            if len(parent) and parent[-1].tag in self.LIST_TYPES:
                self.parser.parseBlocks(parent[-1], [block])
            else:
                self.parser.parseBlocks(parent, [block])
        elif sibling.tag in self.ITEM_TYPES:
            self.parser.parseBlocks(sibling, [block])
        elif len(sibling) and sibling[-1].tag in self.ITEM_TYPES:
            if sibling[-1].text:
                p = etree.Element('p')
                p.text = sibling[-1].text
                sibling[-1].text = ''
                sibling[-1].insert(0, p)
            self.parser.parseChunk(sibling[-1], block)
        else:
            self.create_item(sibling, block)
        self.parser.state.reset()

    def create_item(self, parent: etree.Element, block: str) -> None:
        """ Create a new `li` and parse the block with it as the parent. """
        li = etree.SubElement(parent, 'li')
        self.parser.parseBlocks(li, [block])

    def get_level(self, parent: etree.Element, block: str) -> tuple[int, etree.Element]:
        """ Get level of indentation based on list level. """
        m = self.INDENT_RE.match(block)
        if m:
            indent_level = len(m.group(1)) / self.tab_length
        else:
            indent_level = 0
        if self.parser.state.isstate('list'):
            level = 1
        else:
            level = 0
        while indent_level > level:
            child = self.lastChild(parent)
            if child is not None and (child.tag in self.LIST_TYPES or child.tag in self.ITEM_TYPES):
                if child.tag in self.LIST_TYPES:
                    level += 1
                parent = child
            else:
                break
        return (level, parent)