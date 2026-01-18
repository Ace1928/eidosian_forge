import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_right_arrow_at_end(self):
    pos = len(self._line)
    expected = (pos, self._line)
    result = right_arrow(pos, self._line)
    self.assertEqual(expected, result)