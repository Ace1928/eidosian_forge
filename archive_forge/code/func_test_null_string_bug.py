from __future__ import unicode_literals
import unittest
import commonmark
from commonmark.blocks import Parser
from commonmark.render.html import HtmlRenderer
from commonmark.inlines import InlineParser
from commonmark.node import NodeWalker, Node
def test_null_string_bug(self):
    s = commonmark.commonmark('>     sometext\n>\n\n')
    self.assertEqual(s, '<blockquote>\n<pre><code>sometext\n</code></pre>\n</blockquote>\n')