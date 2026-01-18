import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_transpose_first_character(self):
    self.assertEqual(transpose_character_before_cursor(0, 'a'), (0, 'a'))
    self.assertEqual(transpose_character_before_cursor(0, 'as'), (0, 'as'))