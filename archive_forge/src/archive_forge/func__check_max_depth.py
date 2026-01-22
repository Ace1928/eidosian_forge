import os
import gc
import sys
import queue
import struct
import weakref
import warnings
import itertools
import traceback
import threading
from time import time, sleep
import multiprocessing as mp
from functools import partial
from pickle import PicklingError
from concurrent.futures import Executor
from concurrent.futures._base import LOGGER
from concurrent.futures.process import BrokenProcessPool as _BPPException
from multiprocessing.connection import wait
from ._base import Future
from .backend import get_context
from .backend.context import cpu_count, _MAX_WINDOWS_WORKERS
from .backend.queues import Queue, SimpleQueue
from .backend.reduction import set_loky_pickler, get_loky_pickler_name
from .backend.utils import kill_process_tree, get_exitcodes_terminated_worker
from .initializers import _prepare_initializer
def _check_max_depth(context):
    global _CURRENT_DEPTH
    if context.get_start_method() == 'fork' and _CURRENT_DEPTH > 0:
        raise LokyRecursionError("Could not spawn extra nested processes at depth superior to MAX_DEPTH=1. It is not possible to increase this limit when using the 'fork' start method.")
    if 0 < MAX_DEPTH and _CURRENT_DEPTH + 1 > MAX_DEPTH:
        raise LokyRecursionError(f'Could not spawn extra nested processes at depth superior to MAX_DEPTH={MAX_DEPTH}. If this is intendend, you can change this limit with the LOKY_MAX_DEPTH environment variable.')