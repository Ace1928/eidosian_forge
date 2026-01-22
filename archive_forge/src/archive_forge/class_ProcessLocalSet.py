import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid
from traceback import format_exc
from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
class ProcessLocalSet(set):

    def __init__(self):
        util.register_after_fork(self, lambda obj: obj.clear())

    def __reduce__(self):
        return (type(self), ())