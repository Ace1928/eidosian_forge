from __future__ import annotations
import errno
import os
import signal
from twisted.python.runtime import platformType
from twisted.trial.unittest import SynchronousTestCase

        The file descriptor passed to L{installHandler} has a byte written to
        it when SIGCHLD is delivered to the process.
        