from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
class UListProcessor(OListProcessor):
    """ Process unordered list blocks. """
    TAG: str = 'ul'
    ' The tag used for the the wrapping element. '

    def __init__(self, parser: BlockParser):
        super().__init__(parser)
        self.RE = re.compile('^[ ]{0,%d}[*+-][ ]+(.*)' % (self.tab_length - 1))