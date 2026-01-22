import socket
import sys
import warnings
from typing import Tuple, Type
from zope.interface import implementer
from twisted.internet import base, error, interfaces, main
from twisted.internet._dumbwin32proc import Process
from twisted.internet.iocpreactor import iocpsupport as _iocp, tcp, udp
from twisted.internet.iocpreactor.const import WAIT_TIMEOUT
from twisted.internet.win32eventreactor import _ThreadedWin32EventsMixin
from twisted.python import failure, log
def _callEventCallback(self, rc, numBytes, evt):
    owner = evt.owner
    why = None
    try:
        evt.callback(rc, numBytes, evt)
        handfn = getattr(owner, 'getFileHandle', None)
        if not handfn:
            why = _NO_GETHANDLE
        elif handfn() == -1:
            why = _NO_FILEDESC
        if why:
            return
    except BaseException:
        why = sys.exc_info()[1]
        log.err()
    if why:
        owner.loseConnection(failure.Failure(why))