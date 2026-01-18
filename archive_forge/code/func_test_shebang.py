import os
import sys
from io import StringIO
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SkipTest, TestCase
def test_shebang(self):
    """
        The example scripts start with the standard shebang line.
        """
    with self.examplePath.open() as f:
        self.assertEqual(f.readline().rstrip(), b'#!/usr/bin/env python')