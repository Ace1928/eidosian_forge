from abc import abstractmethod
from contextlib import closing
import functools
import hashlib
import multiprocessing
import multiprocessing.dummy
import os
import queue
import random
import shutil
import sys  # pylint: disable=unused-import
import tarfile
import threading
import time
import typing
import urllib
import weakref
import zipfile
import numpy as np
from tensorflow.python.framework import tensor
from six.moves.urllib.request import urlopen
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
class SequenceEnqueuer(object):
    """Base class to enqueue inputs.

  The task of an Enqueuer is to use parallelism to speed up preprocessing.
  This is done with processes or threads.

  Example:

  ```python
      enqueuer = SequenceEnqueuer(...)
      enqueuer.start()
      datas = enqueuer.get()
      for data in datas:
          # Use the inputs; training, evaluating, predicting.
          # ... stop sometime.
      enqueuer.stop()
  ```

  The `enqueuer.get()` should be an infinite stream of datas.
  """

    def __init__(self, sequence, use_multiprocessing=False):
        self.sequence = sequence
        self.use_multiprocessing = use_multiprocessing
        global _SEQUENCE_COUNTER
        if _SEQUENCE_COUNTER is None:
            try:
                _SEQUENCE_COUNTER = multiprocessing.Value('i', 0)
            except OSError:
                _SEQUENCE_COUNTER = 0
        if isinstance(_SEQUENCE_COUNTER, int):
            self.uid = _SEQUENCE_COUNTER
            _SEQUENCE_COUNTER += 1
        else:
            with _SEQUENCE_COUNTER.get_lock():
                self.uid = _SEQUENCE_COUNTER.value
                _SEQUENCE_COUNTER.value += 1
        self.workers = 0
        self.executor_fn = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None

    def is_running(self):
        return self.stop_signal is not None and (not self.stop_signal.is_set())

    def start(self, workers=1, max_queue_size=10):
        """Starts the handler's workers.

    Args:
        workers: Number of workers.
        max_queue_size: queue size
            (when full, workers could block on `put()`)
    """
        if self.use_multiprocessing:
            self.executor_fn = self._get_executor_init(workers)
        else:
            self.executor_fn = lambda _: get_pool_class(False)(workers)
        self.workers = workers
        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _send_sequence(self):
        """Sends current Iterable to all workers."""
        _SHARED_SEQUENCES[self.uid] = self.sequence

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

    Should be called by the same thread which called `start()`.

    Args:
        timeout: maximum time to wait on `thread.join()`
    """
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.run_thread.join(timeout)
        _SHARED_SEQUENCES[self.uid] = None

    def __del__(self):
        if self.is_running():
            self.stop()

    @abstractmethod
    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        raise NotImplementedError

    @abstractmethod
    def _get_executor_init(self, workers):
        """Gets the Pool initializer for multiprocessing.

    Args:
        workers: Number of workers.

    Returns:
        Function, a Function to initialize the pool
    """
        raise NotImplementedError

    @abstractmethod
    def get(self):
        """Creates a generator to extract data from the queue.

    Skip the data if it is `None`.
    # Returns
        Generator yielding tuples `(inputs, targets)`
            or `(inputs, targets, sample_weights)`.
    """
        raise NotImplementedError