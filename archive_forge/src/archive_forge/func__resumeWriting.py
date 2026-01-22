import errno
from zope.interface import implementer
from twisted.internet import error, interfaces, main
from twisted.internet.abstract import _ConsumerMixin, _dataMustBeBytes, _LogOwner
from twisted.internet.iocpreactor import iocpsupport as _iocp
from twisted.internet.iocpreactor.const import ERROR_HANDLE_EOF, ERROR_IO_PENDING
from twisted.python import failure
def _resumeWriting(self):
    self._writeScheduled = None
    self.doWrite()