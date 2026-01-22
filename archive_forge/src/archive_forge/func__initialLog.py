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
def _initialLog(self):
    """
        Print twistd start log message.
        """
    from twisted.internet import reactor
    logger._loggerFor(self).info('twistd {version} ({exe} {pyVersion}) starting up.', version=copyright.version, exe=sys.executable, pyVersion=runtime.shortPythonVersion())
    logger._loggerFor(self).info('reactor class: {reactor}.', reactor=qual(reactor.__class__))