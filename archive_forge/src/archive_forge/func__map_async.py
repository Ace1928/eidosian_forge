import collections
import copy
import gc
import itertools
import logging
import os
import queue
import sys
import threading
import time
from multiprocessing import TimeoutError
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple
import ray
from ray._private.usage import usage_lib
from ray.util import log_once
def _map_async(self, func, iterable, chunksize=None, unpack_args=False, callback=None, error_callback=None):
    self._check_running()
    object_refs = self._chunk_and_run(func, iterable, chunksize=chunksize, unpack_args=unpack_args)
    return AsyncResult(object_refs, callback, error_callback)