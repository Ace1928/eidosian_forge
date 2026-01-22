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

        Configure global process state based on the given application and run
        the application.

        @param application: An object which can be adapted to
            L{service.IProcess} and L{service.IService}.
        