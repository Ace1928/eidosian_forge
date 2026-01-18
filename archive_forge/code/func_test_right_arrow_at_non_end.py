import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_right_arrow_at_non_end(self):
    for i in range(len(self._line) - 1):
        expected = (i + 1, self._line)
        result = right_arrow(i, self._line)
        self.assertEqual(expected, result)