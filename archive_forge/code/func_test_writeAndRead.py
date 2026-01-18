import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
def test_writeAndRead(self):
    """
        Test that the number of bytes L{fdesc.writeToFD} reports as written
        with its return value are seen by L{fdesc.readFromFD}.
        """
    n = self.write(b'hello')
    self.assertTrue(n > 0)
    s = self.read()
    self.assertEqual(len(s), n)
    self.assertEqual(b'hello'[:n], s)