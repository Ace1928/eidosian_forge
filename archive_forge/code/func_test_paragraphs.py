from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import markup
def test_paragraphs(self):
    self.assert_item_types('      Hello world\n\n      Hello wold', [markup.CommentType.PARAGRAPH, markup.CommentType.SEPARATOR, markup.CommentType.PARAGRAPH])