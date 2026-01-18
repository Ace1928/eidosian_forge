import os
import sys
from io import StringIO
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SkipTest, TestCase

        The example script prints a usage message to stderr if it is
        passed unrecognized command line arguments.

        The first line should contain a USAGE summary, explaining the
        accepted command arguments.

        The last line should contain an ERROR summary, explaining that
        incorrect arguments were supplied.
        