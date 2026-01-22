from concurrent.futures import _base
import itertools
import queue
import threading
import types
import weakref
import os
class BrokenThreadPool(_base.BrokenExecutor):
    """
    Raised when a worker thread in a ThreadPoolExecutor failed initializing.
    """