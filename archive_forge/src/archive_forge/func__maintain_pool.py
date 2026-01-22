import collections
import itertools
import os
import queue
import threading
import time
import traceback
import types
import warnings
from . import util
from . import get_context, TimeoutError
from .connection import wait
@staticmethod
def _maintain_pool(ctx, Process, processes, pool, inqueue, outqueue, initializer, initargs, maxtasksperchild, wrap_exception):
    """Clean up any exited workers and start replacements for them.
        """
    if Pool._join_exited_workers(pool):
        Pool._repopulate_pool_static(ctx, Process, processes, pool, inqueue, outqueue, initializer, initargs, maxtasksperchild, wrap_exception)