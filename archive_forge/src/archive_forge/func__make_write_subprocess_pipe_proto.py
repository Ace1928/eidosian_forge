import collections
import subprocess
import sys
import warnings
from . import futures
from . import protocols
from . import transports
from .coroutines import coroutine
from .log import logger
def _make_write_subprocess_pipe_proto(self, fd):
    raise NotImplementedError