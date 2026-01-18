import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
def test_writeErrors(self):
    """
        Test error path for L{fdesc.writeTod}.
        """
    oldOsWrite = os.write

    def eagainWrite(fd, data):
        err = OSError()
        err.errno = errno.EAGAIN
        raise err
    os.write = eagainWrite
    try:
        self.assertEqual(self.write(b's'), 0)
    finally:
        os.write = oldOsWrite

    def eintrWrite(fd, data):
        err = OSError()
        err.errno = errno.EINTR
        raise err
    os.write = eintrWrite
    try:
        self.assertEqual(self.write(b's'), 0)
    finally:
        os.write = oldOsWrite