import errno
import os
import pwd
import sys
import traceback
from twisted import copyright, logger
from twisted.application import app, service
from twisted.internet.interfaces import IReactorDaemonize
from twisted.python import log, logfile, usage
from twisted.python.runtime import platformType
from twisted.python.util import gidFromString, switchUID, uidFromString, untilConcludes
class UnixAppLogger(app.AppLogger):
    """
    A logger able to log to syslog, to files, and to stdout.

    @ivar _syslog: A flag indicating whether to use syslog instead of file
        logging.
    @type _syslog: C{bool}

    @ivar _syslogPrefix: If C{sysLog} is C{True}, the string prefix to use for
        syslog messages.
    @type _syslogPrefix: C{str}

    @ivar _nodaemon: A flag indicating the process will not be daemonizing.
    @type _nodaemon: C{bool}
    """

    def __init__(self, options):
        app.AppLogger.__init__(self, options)
        self._syslog = options.get('syslog', False)
        self._syslogPrefix = options.get('prefix', '')
        self._nodaemon = options.get('nodaemon', False)

    def _getLogObserver(self):
        """
        Create and return a suitable log observer for the given configuration.

        The observer will go to syslog using the prefix C{_syslogPrefix} if
        C{_syslog} is true.  Otherwise, it will go to the file named
        C{_logfilename} or, if C{_nodaemon} is true and C{_logfilename} is
        C{"-"}, to stdout.

        @return: An object suitable to be passed to C{log.addObserver}.
        """
        if self._syslog:
            from twisted.python import syslog
            return syslog.SyslogObserver(self._syslogPrefix).emit
        if self._logfilename == '-':
            if not self._nodaemon:
                sys.exit('Daemons cannot log to stdout, exiting!')
            logFile = sys.stdout
        elif self._nodaemon and (not self._logfilename):
            logFile = sys.stdout
        else:
            if not self._logfilename:
                self._logfilename = 'twistd.log'
            logFile = logfile.LogFile.fromFullPath(self._logfilename)
            try:
                import signal
            except ImportError:
                pass
            else:
                if not signal.getsignal(signal.SIGUSR1):

                    def rotateLog(signal, frame):
                        from twisted.internet import reactor
                        reactor.callFromThread(logFile.rotate)
                    signal.signal(signal.SIGUSR1, rotateLog)
        return logger.textFileLogObserver(logFile)