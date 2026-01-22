import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
class TestHelpers(LineTestCase):

    def test_I(self):
        self.assertEqual(cursor('asd|fgh'), (3, 'asdfgh'))

    def test_decode(self):
        self.assertEqual(decode('a<bd|c>d'), ((3, 'abdcd'), LinePart(1, 4, 'bdc')))
        self.assertEqual(decode('a|<bdc>d'), ((1, 'abdcd'), LinePart(1, 4, 'bdc')))
        self.assertEqual(decode('a<bdc>d|'), ((5, 'abdcd'), LinePart(1, 4, 'bdc')))

    def test_encode(self):
        self.assertEqual(encode(3, 'abdcd', LinePart(1, 4, 'bdc')), 'a<bd|c>d')
        self.assertEqual(encode(1, 'abdcd', LinePart(1, 4, 'bdc')), 'a|<bdc>d')
        self.assertEqual(encode(4, 'abdcd', LinePart(1, 4, 'bdc')), 'a<bdc|>d')
        self.assertEqual(encode(5, 'abdcd', LinePart(1, 4, 'bdc')), 'a<bdc>d|')

    def test_assert_access(self):

        def dumb_func(cursor_offset, line):
            return LinePart(0, 2, 'ab')
        self.func = dumb_func
        self.assertAccess('<a|b>d')