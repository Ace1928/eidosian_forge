import itertools
import os
import sys
from unittest import skipIf
from twisted.internet import defer, error, protocol, reactor, stdio
from twisted.python import filepath, log
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SkipTest, TestCase
def test_producer(self):
    """
        Verify that the transport of a protocol connected to L{StandardIO}
        is a working L{IProducer} provider.
        """
    p = StandardIOTestProcessProtocol()
    d = p.onCompletion
    written = []
    toWrite = list(range(100))

    def connectionMade(ign):
        if toWrite:
            written.append(b'%d\n' % (toWrite.pop(),))
            proc.write(written[-1])
            reactor.callLater(0.01, connectionMade, None)
    proc = self._spawnProcess(p, b'stdio_test_producer')
    p.onConnection.addCallback(connectionMade)

    def processEnded(reason):
        self.assertEqual(p.data[1], b''.join(written))
        self.assertFalse(toWrite, 'Connection lost with %d writes left to go.' % (len(toWrite),))
        reason.trap(error.ProcessDone)
    return self._requireFailure(d, processEnded)