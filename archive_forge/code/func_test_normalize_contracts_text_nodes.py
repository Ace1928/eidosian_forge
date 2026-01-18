from __future__ import unicode_literals
import unittest
import commonmark
from commonmark.blocks import Parser
from commonmark.render.html import HtmlRenderer
from commonmark.inlines import InlineParser
from commonmark.node import NodeWalker, Node
def test_normalize_contracts_text_nodes(self):
    md = '_a'
    ast = Parser().parse(md)

    def assert_text_literals(text_literals):
        walker = ast.walker()
        document, _ = walker.next()
        self.assertEqual(document.t, 'document')
        paragraph, _ = walker.next()
        self.assertEqual(paragraph.t, 'paragraph')
        for literal in text_literals:
            text, _ = walker.next()
            self.assertEqual(text.t, 'text')
            self.assertEqual(text.literal, literal)
        paragraph, _ = walker.next()
        self.assertEqual(paragraph.t, 'paragraph')
    assert_text_literals(['_', 'a'])
    ast.normalize()
    assert_text_literals(['_a'])
    ast.normalize()
    assert_text_literals(['_a'])