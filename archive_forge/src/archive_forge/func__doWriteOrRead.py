import errno
import select
from zope.interface import Attribute, Interface, declarations, implementer
from twisted.internet import main, posixbase
from twisted.internet.interfaces import IReactorDaemonize, IReactorFDSet
from twisted.python import failure, log
def _doWriteOrRead(self, selectable, fd, event):
    """
        Private method called when a FD is ready for reading, writing or was
        lost. Do the work and raise errors where necessary.
        """
    why = None
    inRead = False
    filter, flags, data, fflags = (event.filter, event.flags, event.data, event.fflags)
    if flags & KQ_EV_EOF and data and fflags:
        why = main.CONNECTION_LOST
    else:
        try:
            if selectable.fileno() == -1:
                inRead = False
                why = posixbase._NO_FILEDESC
            else:
                if filter == KQ_FILTER_READ:
                    inRead = True
                    why = selectable.doRead()
                if filter == KQ_FILTER_WRITE:
                    inRead = False
                    why = selectable.doWrite()
        except BaseException:
            why = failure.Failure()
            log.err(why, 'An exception was raised from application code while processing a reactor selectable')
    if why:
        self._disconnectSelectable(selectable, why, inRead)