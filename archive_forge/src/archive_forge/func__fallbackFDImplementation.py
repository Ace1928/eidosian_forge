from __future__ import annotations
import errno
import gc
import io
import os
import signal
import stat
import sys
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from zope.interface import implementer
from twisted.internet import abstract, error, fdesc
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IProcessTransport
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform
from twisted.python.util import switchUID
def _fallbackFDImplementation(self):
    """
        Fallback implementation where either the resource module can inform us
        about the upper bound of how many FDs to expect, or where we just guess
        a constant maximum if there is no resource module.

        All possible file descriptors from 0 to that upper bound are returned
        with no attempt to exclude invalid file descriptor values.
        """
    try:
        import resource
    except ImportError:
        maxfds = 1024
    else:
        maxfds = min(1024, resource.getrlimit(resource.RLIMIT_NOFILE)[1])
    return range(maxfds)