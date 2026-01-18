import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_seq(self):

    def f(cursor_offset, line):
        return ('hi', 2)
    self.edits.add('a', f)
    self.assertIn('a', self.edits)
    self.assertEqual(self.edits['a'], f)
    self.assertEqual(self.edits.call('a', cursor_offset=3, line='hello'), ('hi', 2))
    with self.assertRaises(KeyError):
        self.edits['b']
    with self.assertRaises(KeyError):
        self.edits.call('b')