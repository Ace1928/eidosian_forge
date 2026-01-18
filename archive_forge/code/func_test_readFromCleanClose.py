import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
def test_readFromCleanClose(self):
    """
        Test that using L{fdesc.readFromFD} on a cleanly closed file descriptor
        returns a connection done indicator.
        """
    os.close(self.w)
    self.assertEqual(self.read(), fdesc.CONNECTION_DONE)