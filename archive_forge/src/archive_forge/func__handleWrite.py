import errno
from zope.interface import implementer
from twisted.internet import error, interfaces, main
from twisted.internet.abstract import _ConsumerMixin, _dataMustBeBytes, _LogOwner
from twisted.internet.iocpreactor import iocpsupport as _iocp
from twisted.internet.iocpreactor.const import ERROR_HANDLE_EOF, ERROR_IO_PENDING
from twisted.python import failure
def _handleWrite(self, rc, numBytesWritten, evt):
    """
        Returns false if we should stop writing for now
        """
    if self.disconnected or self._writeDisconnected:
        return False
    if rc:
        self.connectionLost(failure.Failure(error.ConnectionLost('write error -- %s (%s)' % (errno.errorcode.get(rc, 'unknown'), rc))))
        return False
    else:
        self.offset += numBytesWritten
        if self.offset == len(self.dataBuffer) and (not self._tempDataLen):
            self.dataBuffer = b''
            self.offset = 0
            self.stopWriting()
            if self.producer is not None and (not self.streamingProducer or self.producerPaused):
                self.producerPaused = True
                self.producer.resumeProducing()
            elif self.disconnecting:
                self.connectionLost(failure.Failure(main.CONNECTION_DONE))
            elif self._writeDisconnecting:
                self._writeDisconnected = True
                self._closeWriteConnection()
            return False
        else:
            return True