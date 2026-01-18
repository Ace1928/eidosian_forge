import errno
import sys
from io import BytesIO
from twisted.internet.testing import StringTransport
from twisted.protocols.amp import AMP
from twisted.trial._dist import (
from twisted.trial._dist.workertrial import WorkerLogObserver, main
from twisted.trial.unittest import TestCase
def test_readInterrupted(self):
    """
        If reading the input stream fails with a C{IOError} with errno
        C{EINTR}, L{main} ignores it and continues reading.
        """
    excInfos = []

    class FakeStream:
        count = 0

        def read(oself, size):
            oself.count += 1
            if oself.count == 1:
                raise OSError(errno.EINTR)
            else:
                excInfos.append(sys.exc_info())
            return b''
    self.readStream = FakeStream()
    main(self.fdopen)
    self.assertEqual(b'', self.writeStream.getvalue())
    self.assertEqual([(None, None, None)], excInfos)