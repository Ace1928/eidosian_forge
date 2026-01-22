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
def _reportImportError(self, module, e):
    """
        Helper method to report an import error with a profile module. This
        has to be explicit because some of these modules are removed by
        distributions due to them being non-free.
        """
    s = f'Failed to import module {module}: {e}'
    s += '\nThis is most likely caused by your operating system not including\nthe module due to it being non-free. Either do not use the option\n--profile, or install the module; your operating system vendor\nmay provide it in a separate package.\n'
    raise SystemExit(s)