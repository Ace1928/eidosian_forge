import errno
import sys
from io import BytesIO
from twisted.internet.testing import StringTransport
from twisted.protocols.amp import AMP
from twisted.trial._dist import (
from twisted.trial._dist.workertrial import WorkerLogObserver, main
from twisted.trial.unittest import TestCase
def test_otherReadError(self):
    """
        L{main} only ignores C{IOError} with C{EINTR} errno: otherwise, the
        error pops out.
        """

    class FakeStream:
        count = 0

        def read(oself, size):
            oself.count += 1
            if oself.count == 1:
                raise OSError('Something else')
            return ''
    self.readStream = FakeStream()
    self.assertRaises(IOError, main, self.fdopen)