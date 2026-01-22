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
def _submit_chunk(self, func, iterator, chunksize, actor_index, unpack_args=False):
    chunk = []
    while len(chunk) < chunksize:
        try:
            args = next(iterator)
            if not unpack_args:
                args = (args,)
            chunk.append((args, {}))
        except StopIteration:
            break
    assert len(chunk) > 0
    return self._run_batch(actor_index, func, chunk)