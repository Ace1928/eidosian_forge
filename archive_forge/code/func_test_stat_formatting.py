import unittest
from io import BytesIO, StringIO
from testtools.compat import _b
import subunit
def test_stat_formatting(self):
    expected = '\nTotal tests:       5\nPassed tests:      2\nFailed tests:      2\nSkipped tests:     1\nSeen tags: global, local\n'[1:]
    self.setUpUsedStream()
    self.result.formatStats()
    self.assertEqual(expected, self.output.getvalue())