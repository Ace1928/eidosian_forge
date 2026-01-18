import sys
import unittest
def test_simple_w_CRLF(self):
    LINES = ['Three blind mice', 'See how they run']
    text = '\r\n'.join(LINES)
    self.assertEqual(self._callFUT(text, 1), '\n'.join([' ' + line for line in LINES]))