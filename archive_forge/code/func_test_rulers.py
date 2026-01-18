from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import markup
def test_rulers(self):
    self.assert_item_types('      --------------------\n      This is some\n      text that I expect\n      to reflow\n      --------------------\n      ', [markup.CommentType.RULER, markup.CommentType.PARAGRAPH, markup.CommentType.RULER, markup.CommentType.SEPARATOR])