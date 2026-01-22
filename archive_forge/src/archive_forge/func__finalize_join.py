import sys
import os
import threading
import collections
import time
import types
import weakref
import errno
from queue import Empty, Full
from . import connection
from . import context
from .util import debug, info, Finalize, register_after_fork, is_exiting
@staticmethod
def _finalize_join(twr):
    debug('joining queue thread')
    thread = twr()
    if thread is not None:
        thread.join()
        debug('... queue thread joined')
    else:
        debug('... queue thread already dead')