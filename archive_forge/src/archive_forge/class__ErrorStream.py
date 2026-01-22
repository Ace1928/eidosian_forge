from collections.abc import Sequence
from sys import exc_info
from warnings import warn
from zope.interface import implementer
from twisted.internet.threads import blockingCallFromThread
from twisted.logger import Logger
from twisted.python.failure import Failure
from twisted.web.http import INTERNAL_SERVER_ERROR
from twisted.web.resource import IResource
from twisted.web.server import NOT_DONE_YET
class _ErrorStream:
    """
    File-like object instances of which are used as the value for the
    C{'wsgi.errors'} key in the C{environ} dictionary passed to the application
    object.

    This simply passes writes on to L{logging<twisted.logger>} system as
    error events from the C{'wsgi'} system.  In the future, it may be desirable
    to expose more information in the events it logs, such as the application
    object which generated the message.
    """
    _log = Logger()

    def write(self, data):
        """
        Generate an event for the logging system with the given bytes as the
        message.

        This is called in a WSGI application thread, not the I/O thread.

        @type data: str

        @raise TypeError: On Python 3, if C{data} is not a native string. On
            Python 2 a warning will be issued.
        """
        if not isinstance(data, str):
            if str is bytes:
                warn('write() argument should be str, not %r (%s)' % (data, type(data).__name__), category=UnicodeWarning)
            else:
                raise TypeError('write() argument must be str, not %r (%s)' % (data, type(data).__name__))
        self._log.error(data, system='wsgi', isError=True, message=(data,))

    def writelines(self, iovec):
        """
        Join the given lines and pass them to C{write} to be handled in the
        usual way.

        This is called in a WSGI application thread, not the I/O thread.

        @param iovec: A C{list} of C{'\\n'}-terminated C{str} which will be
            logged.

        @raise TypeError: On Python 3, if C{iovec} contains any non-native
            strings. On Python 2 a warning will be issued.
        """
        self.write(''.join(iovec))

    def flush(self):
        """
        Nothing is buffered, so flushing does nothing.  This method is required
        to exist by PEP 333, though.

        This is called in a WSGI application thread, not the I/O thread.
        """