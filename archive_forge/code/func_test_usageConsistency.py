import os
import sys
from io import StringIO
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SkipTest, TestCase
def test_usageConsistency(self):
    """
        The example script prints a usage message to stdout if it is
        passed a --help option and then exits.

        The first line should contain a USAGE summary, explaining the
        accepted command arguments.
        """
    self.assertRaises(SystemExit, self.example.main, None, '--help')
    out = self.fakeOut.getvalue().splitlines()
    self.assertTrue(out[0].startswith('Usage:'), 'Usage message first line should start with "Usage:". Actual: %r' % (out[0],))