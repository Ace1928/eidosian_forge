from __future__ import absolute_import, unicode_literals
import re
from commonmark import common
from commonmark.common import unescape_string
from commonmark.inlines import InlineParser
from commonmark.node import Node
class ThematicBreak(Block):
    accepts_lines = False

    @staticmethod
    def continue_(parser=None, container=None):
        return 1

    @staticmethod
    def finalize(parser=None, block=None):
        return

    @staticmethod
    def can_contain(t):
        return False