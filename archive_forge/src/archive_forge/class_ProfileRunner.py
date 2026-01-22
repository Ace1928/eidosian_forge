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
class ProfileRunner(_BasicProfiler):
    """
    Runner for the standard profile module.
    """

    def run(self, reactor):
        """
        Run reactor under the standard profiler.
        """
        try:
            import profile
        except ImportError as e:
            self._reportImportError('profile', e)
        p = profile.Profile()
        p.runcall(reactor.run)
        if self.saveStats:
            p.dump_stats(self.profileOutput)
        else:
            tmp, sys.stdout = (sys.stdout, open(self.profileOutput, 'a'))
            try:
                p.print_stats()
            finally:
                sys.stdout, tmp = (tmp, sys.stdout)
                tmp.close()