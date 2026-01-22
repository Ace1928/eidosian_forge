import socket
from incremental import Version
from twisted.python import deprecate
class ProcessTerminated(ConnectionLost):
    __doc__ = MESSAGE = '\n    A process has ended with a probable error condition\n\n    @ivar exitCode: See L{__init__}\n    @ivar signal: See L{__init__}\n    @ivar status: See L{__init__}\n    '

    def __init__(self, exitCode=None, signal=None, status=None):
        """
        @param exitCode: The exit status of the process.  This is roughly like
            the value you might pass to L{os._exit}.  This is L{None} if the
            process exited due to a signal.
        @type exitCode: L{int} or L{None}

        @param signal: The exit signal of the process.  This is L{None} if the
            process did not exit due to a signal.
        @type signal: L{int} or L{None}

        @param status: The exit code of the process.  This is a platform
            specific combination of the exit code and the exit signal.  See
            L{os.WIFEXITED} and related functions.
        @type status: L{int}
        """
        self.exitCode = exitCode
        self.signal = signal
        self.status = status
        s = 'process ended'
        if exitCode is not None:
            s = s + ' with exit code %s' % exitCode
        if signal is not None:
            s = s + ' by signal %s' % signal
        Exception.__init__(self, s)