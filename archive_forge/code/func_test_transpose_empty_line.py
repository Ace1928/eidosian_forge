import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_transpose_empty_line(self):
    self.assertEqual(transpose_character_before_cursor(0, ''), (0, ''))