import sys
import unittest
def test_with_munge(self):
    TEXT = 'This is a piece of text longer than 15 characters, \nand split across multiple lines.'
    EXPECTED = '  This is a piece\n  of text longer\n  than 15 characters,\n  and split across\n  multiple lines.\n '
    self.assertEqual(self._callFUT(TEXT, 1, munge=1, width=15), EXPECTED)