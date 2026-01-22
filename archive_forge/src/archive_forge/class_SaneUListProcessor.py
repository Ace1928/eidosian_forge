from __future__ import annotations
from . import Extension
from ..blockprocessors import OListProcessor, UListProcessor
import re
from typing import TYPE_CHECKING
class SaneUListProcessor(UListProcessor):
    """ Override `SIBLING_TAGS` to not include `ol`. """
    SIBLING_TAGS = ['ul']
    ' Exclude `ol` from list of siblings. '

    def __init__(self, parser: blockparser.BlockParser):
        super().__init__(parser)
        self.CHILD_RE = re.compile('^[ ]{0,%d}(([*+-]))[ ]+(.*)' % (self.tab_length - 1))