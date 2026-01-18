import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
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