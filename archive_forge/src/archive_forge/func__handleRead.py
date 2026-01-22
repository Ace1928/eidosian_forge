import errno
from zope.interface import implementer
from twisted.internet import error, interfaces, main
from twisted.internet.abstract import _ConsumerMixin, _dataMustBeBytes, _LogOwner
from twisted.internet.iocpreactor import iocpsupport as _iocp
from twisted.internet.iocpreactor.const import ERROR_HANDLE_EOF, ERROR_IO_PENDING
from twisted.python import failure
def _handleRead(self, rc, data, evt):
    """
        Returns False if we should stop reading for now
        """
    if self.disconnected:
        return False
    if not (rc or data) or rc in (errno.WSAEDISCON, ERROR_HANDLE_EOF):
        self.reactor.removeActiveHandle(self)
        self.readConnectionLost(failure.Failure(main.CONNECTION_DONE))
        return False
    elif rc:
        self.connectionLost(failure.Failure(error.ConnectionLost('read error -- %s (%s)' % (errno.errorcode.get(rc, 'unknown'), rc))))
        return False
    else:
        assert self._readSize == 0
        assert self._readNextBuffer == 0
        self._readSize = data
        return self._dispatchData()