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
class RemoteError(Exception):

    def __str__(self):
        return '\n' + '-' * 75 + '\n' + str(self.args[0]) + '-' * 75