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
class OrderedIMapIterator(IMapIterator):
    """Iterator to the results of tasks submitted using `imap`.

    The results are returned in the same order that they were submitted, even
    if they don't finish in that order. Only one batch of tasks per actor
    process is submitted at a time - the rest are submitted as results come in.

    Should not be constructed directly.
    """

    def next(self, timeout=None):
        if len(self._ready_objects) == 0:
            if self._finished_iterating and self._next_chunk_index == len(self._submitted_chunks):
                raise StopIteration
            index = -1
            while index != self._next_chunk_index:
                start = time.time()
                index = self._result_thread.next_ready_index(timeout=timeout)
                self._submit_next_chunk()
                self._submitted_chunks[index] = True
                if timeout is not None:
                    timeout = max(0, timeout - (time.time() - start))
            while self._next_chunk_index < len(self._submitted_chunks) and self._submitted_chunks[self._next_chunk_index]:
                for result in self._result_thread.result(self._next_chunk_index):
                    self._ready_objects.append(result)
                self._next_chunk_index += 1
        return self._ready_objects.popleft()