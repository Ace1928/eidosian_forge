from __future__ import unicode_literals
import unittest
import commonmark
from commonmark.blocks import Parser
from commonmark.render.html import HtmlRenderer
from commonmark.inlines import InlineParser
from commonmark.node import NodeWalker, Node
@given(text())
def test_random_text(self, s):
    commonmark.commonmark(s)