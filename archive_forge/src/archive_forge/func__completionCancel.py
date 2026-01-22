import os
import tempfile
from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.mail import smtp
from twisted.mail.interfaces import IAlias
from twisted.python import failure, log
def _completionCancel(self):
    """
        Handle the expiration of the timeout for the child process to exit by
        terminating the child process forcefully and issuing a failure to the
        L{completion} deferred.
        """
    self._timeoutCallID = None
    self.protocol.transport.signalProcess('KILL')
    exc = ProcessAliasTimeout(f'No answer after {self.completionTimeout} seconds')
    self.protocol.onEnd = None
    self.completion.errback(failure.Failure(exc))