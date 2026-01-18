import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
def test_setCloseOnExec(self):
    """
        A file descriptor passed to L{fdesc._setCloseOnExec} is not inherited
        by a new process image created with one of the exec family of
        functions.
        """
    with open(self.mktemp(), 'wb') as fObj:
        fdesc._setCloseOnExec(fObj.fileno())
        status = self._execWithFileDescriptor(fObj)
        self.assertTrue(os.WIFEXITED(status))
        self.assertEqual(os.WEXITSTATUS(status), 0)