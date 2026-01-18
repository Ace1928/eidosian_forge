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
def track_segment(self, c, segment_name):
    """Adds the supplied shared memory block name to Server's tracker."""
    self.shared_memory_context.register_segment(segment_name)