import socket
import sys
from typing import Sequence
from zope.interface import classImplements, implementer
from twisted.internet import error, tcp, udp
from twisted.internet.base import ReactorBase
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform, platformType
from ._signals import (
class _PollLikeMixin:
    """
    Mixin for poll-like reactors.

    Subclasses must define the following attributes::

      - _POLL_DISCONNECTED - Bitmask for events indicating a connection was
        lost.
      - _POLL_IN - Bitmask for events indicating there is input to read.
      - _POLL_OUT - Bitmask for events indicating output can be written.

    Must be mixed in to a subclass of PosixReactorBase (for
    _disconnectSelectable).
    """

    def _doReadOrWrite(self, selectable, fd, event):
        """
        fd is available for read or write, do the work and raise errors if
        necessary.
        """
        why = None
        inRead = False
        if event & self._POLL_DISCONNECTED and (not event & self._POLL_IN):
            if fd in self._reads:
                inRead = True
                why = CONNECTION_DONE
            else:
                why = CONNECTION_LOST
        else:
            try:
                if selectable.fileno() == -1:
                    why = _NO_FILEDESC
                else:
                    if event & self._POLL_IN:
                        why = selectable.doRead()
                        inRead = True
                    if not why and event & self._POLL_OUT:
                        why = selectable.doWrite()
                        inRead = False
            except BaseException:
                why = sys.exc_info()[1]
                log.err()
        if why:
            self._disconnectSelectable(selectable, why, inRead)