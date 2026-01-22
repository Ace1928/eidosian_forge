import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
class ReadWriteTests(unittest.SynchronousTestCase):
    """
    Tests for L{fdesc.readFromFD}, L{fdesc.writeToFD}.
    """

    def setUp(self):
        """
        Create a non-blocking pipe that can be used in tests.
        """
        self.r, self.w = os.pipe()
        fdesc.setNonBlocking(self.r)
        fdesc.setNonBlocking(self.w)

    def tearDown(self):
        """
        Close pipes.
        """
        try:
            os.close(self.w)
        except OSError:
            pass
        try:
            os.close(self.r)
        except OSError:
            pass

    def write(self, d):
        """
        Write data to the pipe.
        """
        return fdesc.writeToFD(self.w, d)

    def read(self):
        """
        Read data from the pipe.
        """
        l = []
        res = fdesc.readFromFD(self.r, l.append)
        if res is None:
            if l:
                return l[0]
            else:
                return b''
        else:
            return res

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

    def test_writeAndReadLarge(self):
        """
        Similar to L{test_writeAndRead}, but use a much larger string to verify
        the behavior for that case.
        """
        orig = b'0123456879' * 10000
        written = self.write(orig)
        self.assertTrue(written > 0)
        result = []
        resultlength = 0
        i = 0
        while resultlength < written or i < 50:
            result.append(self.read())
            resultlength += len(result[-1])
            i += 1
        result = b''.join(result)
        self.assertEqual(len(result), written)
        self.assertEqual(orig[:written], result)

    def test_readFromEmpty(self):
        """
        Verify that reading from a file descriptor with no data does not raise
        an exception and does not result in the callback function being called.
        """
        l = []
        result = fdesc.readFromFD(self.r, l.append)
        self.assertEqual(l, [])
        self.assertIsNone(result)

    def test_readFromCleanClose(self):
        """
        Test that using L{fdesc.readFromFD} on a cleanly closed file descriptor
        returns a connection done indicator.
        """
        os.close(self.w)
        self.assertEqual(self.read(), fdesc.CONNECTION_DONE)

    def test_writeToClosed(self):
        """
        Verify that writing with L{fdesc.writeToFD} when the read end is closed
        results in a connection lost indicator.
        """
        os.close(self.r)
        self.assertEqual(self.write(b's'), fdesc.CONNECTION_LOST)

    def test_readFromInvalid(self):
        """
        Verify that reading with L{fdesc.readFromFD} when the read end is
        closed results in a connection lost indicator.
        """
        os.close(self.r)
        self.assertEqual(self.read(), fdesc.CONNECTION_LOST)

    def test_writeToInvalid(self):
        """
        Verify that writing with L{fdesc.writeToFD} when the write end is
        closed results in a connection lost indicator.
        """
        os.close(self.w)
        self.assertEqual(self.write(b's'), fdesc.CONNECTION_LOST)

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