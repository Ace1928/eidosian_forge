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
def _submit_next_chunk(self):
    if self._finished_iterating:
        return
    actor_index = len(self._submitted_chunks) % len(self._pool._actor_pool)
    chunk_iterator = itertools.islice(self._iterator, self._chunksize)
    chunk_list = list(chunk_iterator)
    if len(chunk_list) < self._chunksize:
        self._finished_iterating = True
        if len(chunk_list) == 0:
            return
    chunk_iterator = iter(chunk_list)
    new_chunk_id = self._pool._submit_chunk(self._func, chunk_iterator, self._chunksize, actor_index)
    self._submitted_chunks.append(False)
    self._result_thread.add_object_ref(new_chunk_id)
    if self._finished_iterating:
        self._result_thread.add_object_ref(ResultThread.END_SENTINEL)