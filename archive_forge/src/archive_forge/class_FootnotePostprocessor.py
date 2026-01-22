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
class FootnotePostprocessor(Postprocessor):
    """ Replace placeholders with html entities. """

    def __init__(self, footnotes: FootnoteExtension):
        self.footnotes = footnotes

    def run(self, text: str) -> str:
        text = text.replace(FN_BACKLINK_TEXT, self.footnotes.getConfig('BACKLINK_TEXT'))
        return text.replace(NBSP_PLACEHOLDER, '&#160;')