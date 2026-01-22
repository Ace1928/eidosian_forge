import sys
import warnings
from functools import wraps
from io import BytesIO
from twisted.internet import defer, protocol
from twisted.python import failure
class _EverythingGetter(protocol.ProcessProtocol):

    def __init__(self, deferred, stdinBytes=None):
        self.deferred = deferred
        self.outBuf = BytesIO()
        self.errBuf = BytesIO()
        self.outReceived = self.outBuf.write
        self.errReceived = self.errBuf.write
        self.stdinBytes = stdinBytes

    def connectionMade(self):
        if self.stdinBytes is not None:
            self.transport.writeToChild(0, self.stdinBytes)
            self.transport.closeStdin()

    def processEnded(self, reason):
        out = self.outBuf.getvalue()
        err = self.errBuf.getvalue()
        e = reason.value
        code = e.exitCode
        if e.signal:
            self.deferred.errback((out, err, e.signal))
        else:
            self.deferred.callback((out, err, code))