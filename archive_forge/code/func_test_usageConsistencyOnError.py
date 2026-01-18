import os
import sys
from io import StringIO
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SkipTest, TestCase
def test_usageConsistencyOnError(self):
    """
        The example script prints a usage message to stderr if it is
        passed unrecognized command line arguments.

        The first line should contain a USAGE summary, explaining the
        accepted command arguments.

        The last line should contain an ERROR summary, explaining that
        incorrect arguments were supplied.
        """
    self.assertRaises(SystemExit, self.example.main, None, '--unexpected_argument')
    err = self.fakeErr.getvalue().splitlines()
    self.assertTrue(err[0].startswith('Usage:'), 'Usage message first line should start with "Usage:". Actual: %r' % (err[0],))
    self.assertTrue(err[-1].startswith('ERROR:'), 'Usage message last line should start with "ERROR:" Actual: %r' % (err[-1],))