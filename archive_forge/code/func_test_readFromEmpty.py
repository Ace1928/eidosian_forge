import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
def test_readFromEmpty(self):
    """
        Verify that reading from a file descriptor with no data does not raise
        an exception and does not result in the callback function being called.
        """
    l = []
    result = fdesc.readFromFD(self.r, l.append)
    self.assertEqual(l, [])
    self.assertIsNone(result)