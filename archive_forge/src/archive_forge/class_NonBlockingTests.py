import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
class NonBlockingTests(unittest.SynchronousTestCase):
    """
    Tests for L{fdesc.setNonBlocking} and L{fdesc.setBlocking}.
    """

    def test_setNonBlocking(self):
        """
        L{fdesc.setNonBlocking} sets a file description to non-blocking.
        """
        r, w = os.pipe()
        self.addCleanup(os.close, r)
        self.addCleanup(os.close, w)
        self.assertFalse(fcntl.fcntl(r, fcntl.F_GETFL) & os.O_NONBLOCK)
        fdesc.setNonBlocking(r)
        self.assertTrue(fcntl.fcntl(r, fcntl.F_GETFL) & os.O_NONBLOCK)

    def test_setBlocking(self):
        """
        L{fdesc.setBlocking} sets a file description to blocking.
        """
        r, w = os.pipe()
        self.addCleanup(os.close, r)
        self.addCleanup(os.close, w)
        fdesc.setNonBlocking(r)
        fdesc.setBlocking(r)
        self.assertFalse(fcntl.fcntl(r, fcntl.F_GETFL) & os.O_NONBLOCK)