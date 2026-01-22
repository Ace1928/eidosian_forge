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