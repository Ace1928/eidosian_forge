from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
import xml.etree.ElementTree as etree
import re
from typing import TYPE_CHECKING, Any, Sequence
class TableExtension(Extension):
    """ Add tables to Markdown. """

    def __init__(self, **kwargs):
        self.config = {'use_align_attribute': [False, 'True to use align attribute instead of style.']}
        ' Default configuration options. '
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        """ Add an instance of `TableProcessor` to `BlockParser`. """
        if '|' not in md.ESCAPED_CHARS:
            md.ESCAPED_CHARS.append('|')
        processor = TableProcessor(md.parser, self.getConfigs())
        md.parser.blockprocessors.register(processor, 'table', 75)