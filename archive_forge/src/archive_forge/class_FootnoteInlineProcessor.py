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
class FootnoteInlineProcessor(InlineProcessor):
    """ `InlineProcessor` for footnote markers in a document's body text. """

    def __init__(self, pattern: str, footnotes: FootnoteExtension):
        super().__init__(pattern)
        self.footnotes = footnotes

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        id = m.group(1)
        if id in self.footnotes.footnotes.keys():
            sup = etree.Element('sup')
            a = etree.SubElement(sup, 'a')
            sup.set('id', self.footnotes.makeFootnoteRefId(id, found=True))
            a.set('href', '#' + self.footnotes.makeFootnoteId(id))
            a.set('class', 'footnote-ref')
            a.text = self.footnotes.getConfig('SUPERSCRIPT_TEXT').format(list(self.footnotes.footnotes.keys()).index(id) + 1)
            return (sup, m.start(0), m.end(0))
        else:
            return (None, None, None)