import getpass
import os
import pdb
import signal
import sys
import traceback
import warnings
from operator import attrgetter
from twisted import copyright, logger, plugin
from twisted.application import reactors, service
from twisted.application.reactors import NoSuchReactor, installReactor
from twisted.internet import defer
from twisted.internet.interfaces import _ISupportsExitSignalCapturing
from twisted.persisted import sob
from twisted.python import failure, log, logfile, runtime, usage, util
from twisted.python.reflect import namedAny, namedModule, qual
class AppLogger:
    """
    An L{AppLogger} attaches the configured log observer specified on the
    commandline to a L{ServerOptions} object, a custom L{logger.ILogObserver},
    or a legacy custom {log.ILogObserver}.

    @ivar _logfilename: The name of the file to which to log, if other than the
        default.
    @type _logfilename: C{str}

    @ivar _observerFactory: Callable object that will create a log observer, or
        None.

    @ivar _observer: log observer added at C{start} and removed at C{stop}.
    @type _observer: a callable that implements L{logger.ILogObserver} or
        L{log.ILogObserver}.
    """
    _observer = None

    def __init__(self, options):
        """
        Initialize an L{AppLogger} with a L{ServerOptions}.
        """
        self._logfilename = options.get('logfile', '')
        self._observerFactory = options.get('logger') or None

    def start(self, application):
        """
        Initialize the global logging system for the given application.

        If a custom logger was specified on the command line it will be used.
        If not, and an L{logger.ILogObserver} or legacy L{log.ILogObserver}
        component has been set on C{application}, then it will be used as the
        log observer. Otherwise a log observer will be created based on the
        command line options for built-in loggers (e.g. C{--logfile}).

        @param application: The application on which to check for an
            L{logger.ILogObserver} or legacy L{log.ILogObserver}.
        @type application: L{twisted.python.components.Componentized}
        """
        if self._observerFactory is not None:
            observer = self._observerFactory()
        else:
            observer = application.getComponent(logger.ILogObserver, None)
            if observer is None:
                observer = application.getComponent(log.ILogObserver, None)
        if observer is None:
            observer = self._getLogObserver()
        self._observer = observer
        if logger.ILogObserver.providedBy(self._observer):
            observers = [self._observer]
        elif log.ILogObserver.providedBy(self._observer):
            observers = [logger.LegacyLogObserverWrapper(self._observer)]
        else:
            warnings.warn('Passing a logger factory which makes log observers which do not implement twisted.logger.ILogObserver or twisted.python.log.ILogObserver to twisted.application.app.AppLogger was deprecated in Twisted 16.2. Please use a factory that produces twisted.logger.ILogObserver (or the legacy twisted.python.log.ILogObserver) implementing objects instead.', DeprecationWarning, stacklevel=2)
            observers = [logger.LegacyLogObserverWrapper(self._observer)]
        logger.globalLogBeginner.beginLoggingTo(observers)
        self._initialLog()

    def _initialLog(self):
        """
        Print twistd start log message.
        """
        from twisted.internet import reactor
        logger._loggerFor(self).info('twistd {version} ({exe} {pyVersion}) starting up.', version=copyright.version, exe=sys.executable, pyVersion=runtime.shortPythonVersion())
        logger._loggerFor(self).info('reactor class: {reactor}.', reactor=qual(reactor.__class__))

    def _getLogObserver(self):
        """
        Create a log observer to be added to the logging system before running
        this application.
        """
        if self._logfilename == '-' or not self._logfilename:
            logFile = sys.stdout
        else:
            logFile = logfile.LogFile.fromFullPath(self._logfilename)
        return logger.textFileLogObserver(logFile)

    def stop(self):
        """
        Remove all log observers previously set up by L{AppLogger.start}.
        """
        logger._loggerFor(self).info('Server Shut Down.')
        if self._observer is not None:
            logger.globalLogPublisher.removeObserver(self._observer)
            self._observer = None