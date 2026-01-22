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
class CProfileRunner(_BasicProfiler):
    """
    Runner for the cProfile module.
    """

    def run(self, reactor):
        """
        Run reactor under the cProfile profiler.
        """
        try:
            import cProfile
            import pstats
        except ImportError as e:
            self._reportImportError('cProfile', e)
        p = cProfile.Profile()
        p.runcall(reactor.run)
        if self.saveStats:
            p.dump_stats(self.profileOutput)
        else:
            with open(self.profileOutput, 'w') as stream:
                s = pstats.Stats(p, stream=stream)
                s.strip_dirs()
                s.sort_stats(-1)
                s.print_stats()