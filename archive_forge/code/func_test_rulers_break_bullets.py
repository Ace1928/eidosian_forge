from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import markup
def test_rulers_break_bullets(self):
    self.assert_item_types('      --------------------\n      * Bulleted item\n      * Bulttied item\n      --------------------\n      ', [markup.CommentType.RULER, markup.CommentType.BULLET_LIST, markup.CommentType.RULER, markup.CommentType.SEPARATOR])