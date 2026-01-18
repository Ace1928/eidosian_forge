import errno
import os
import sys
from twisted.internet.protocol import FileWrapper
from twisted.python.log import startLoggingWithObserver, textFromEventDict
from twisted.trial._dist import _WORKER_AMP_STDIN, _WORKER_AMP_STDOUT
from twisted.trial._dist.options import WorkerOptions

        Produce a log output.
        