import sys
import warnings
from functools import wraps
from io import BytesIO
from twisted.internet import defer, protocol
from twisted.python import failure
class _UnexpectedErrorOutput(IOError):
    """
    Standard error data was received where it was not expected.  This is a
    subclass of L{IOError} to preserve backward compatibility with the previous
    error behavior of L{getProcessOutput}.

    @ivar processEnded: A L{Deferred} which will fire when the process which
        produced the data on stderr has ended (exited and all file descriptors
        closed).
    """

    def __init__(self, text, processEnded):
        IOError.__init__(self, f'got stderr: {text!r}')
        self.processEnded = processEnded