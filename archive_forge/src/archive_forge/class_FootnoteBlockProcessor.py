from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..inlinepatterns import InlineProcessor
from ..treeprocessors import Treeprocessor
from ..postprocessors import Postprocessor
from .. import util
from collections import OrderedDict
import re
import copy
import xml.etree.ElementTree as etree
class FootnoteBlockProcessor(BlockProcessor):
    """ Find all footnote references and store for later use. """
    RE = re.compile('^[ ]{0,3}\\[\\^([^\\]]*)\\]:[ ]*(.*)$', re.MULTILINE)

    def __init__(self, footnotes: FootnoteExtension):
        super().__init__(footnotes.parser)
        self.footnotes = footnotes

    def test(self, parent: etree.Element, block: str) -> bool:
        return True

    def run(self, parent: etree.Element, blocks: list[str]) -> bool:
        """ Find, set, and remove footnote definitions. """
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            id = m.group(1)
            fn_blocks = [m.group(2)]
            therest = block[m.end():].lstrip('\n')
            m2 = self.RE.search(therest)
            if m2:
                before = therest[:m2.start()].rstrip('\n')
                fn_blocks[0] = '\n'.join([fn_blocks[0], self.detab(before)]).lstrip('\n')
                blocks.insert(0, therest[m2.start():])
            else:
                fn_blocks[0] = '\n'.join([fn_blocks[0], self.detab(therest)]).strip('\n')
                fn_blocks.extend(self.detectTabbed(blocks))
            footnote = '\n\n'.join(fn_blocks)
            self.footnotes.setFootnote(id, footnote.rstrip())
            if block[:m.start()].strip():
                blocks.insert(0, block[:m.start()].rstrip('\n'))
            return True
        blocks.insert(0, block)
        return False

    def detectTabbed(self, blocks: list[str]) -> list[str]:
        """ Find indented text and remove indent before further processing.

        Returns:
            A list of blocks with indentation removed.
        """
        fn_blocks = []
        while blocks:
            if blocks[0].startswith(' ' * 4):
                block = blocks.pop(0)
                m = self.RE.search(block)
                if m:
                    before = block[:m.start()].rstrip('\n')
                    fn_blocks.append(self.detab(before))
                    blocks.insert(0, block[m.start():])
                    break
                else:
                    fn_blocks.append(self.detab(block))
            else:
                break
        return fn_blocks

    def detab(self, block: str) -> str:
        """ Remove one level of indent from a block.

        Preserve lazily indented blocks by only removing indent from indented lines.
        """
        lines = block.split('\n')
        for i, line in enumerate(lines):
            if line.startswith(' ' * 4):
                lines[i] = line[4:]
        return '\n'.join(lines)