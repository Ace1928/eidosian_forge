import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
def test_readFromInvalid(self):
    """
        Verify that reading with L{fdesc.readFromFD} when the read end is
        closed results in a connection lost indicator.
        """
    os.close(self.r)
    self.assertEqual(self.read(), fdesc.CONNECTION_LOST)