import abc
import atexit
import collections
import functools
import multiprocessing.pool
import threading
import time
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
class SliceAggregator(Aggregator):
    """Combine arrays where the final size is known.

  This class expects to aggregate a single tensor-like rather than a nested
  structure of tensor-likes.

  NumPy copies are an operation that threads handle quite well because all of
  the heavy lifting is in c and does not need the GIL. Moreover, we can perform
  lock-free writes to the same buffer in multiple threads because the nature of
  result aggregation guarantees that either the indices are disjoint or the
  aggregator will throw an exception in finalize. Moreover, because aggregation
  is performed on the slowest varying dimension, assignments for a given batch
  will write to contiguous blocks of memory, further minimizing contention.

  There is, however, some scheduling and context switching overhead which will
  offset the gains from pipelining the slice assignment. Below a given threshold
  it is faster to simply assign in the main thread rather than enqueue the
  assignment in a side thread. The exact threshold will vary from system to
  system, but the time is not very sensitive to the exact transition so a value
  of 2 ** 14 was chosen which should be reasonable on most systems.
  """
    _BINARY_SIZE_THRESHOLD = 2 ** 14
    _MAX_COPY_SECONDS = 300

    def __init__(self, num_samples, batch_size):
        self._async_copies = []
        self._pool = get_copy_pool()
        self._errors = []
        super(SliceAggregator, self).__init__(use_steps=False, num_samples=num_samples, steps=None, batch_size=batch_size)

    def create(self, batch_element):
        shape = (self.num_samples,) + batch_element.shape[1:]
        dtype = batch_element.dtype
        self.results = np.empty(shape=shape, dtype=dtype)

    def aggregate(self, batch_element, batch_start, batch_end):
        if self._errors:
            raise self._errors[0]
        if batch_end - batch_start == self.num_samples:
            if self.num_samples != batch_element.shape[0]:
                raise ValueError('Mismatch between expected batch size and model output batch size. Output shape = {}, expected output shape = shape {}'.format(batch_element.shape, self.results.shape))
            self.results = batch_element
            return
        num_elements = np.prod(batch_element.shape)
        if num_elements < self._BINARY_SIZE_THRESHOLD:
            self.results[batch_start:batch_end] = batch_element
        else:
            is_finished = threading.Event()
            self._pool.apply_async(self._slice_assign, args=(batch_element, batch_start, batch_end, is_finished))
            self._async_copies.append(is_finished)

    def _slice_assign(self, batch_element, batch_start, batch_end, is_finished):
        """Legacy utility method to slice input arrays."""
        try:
            self.results[batch_start:batch_end] = batch_element
        except Exception as e:
            self._errors.append(e)
        finally:
            is_finished.set()

    def finalize(self):
        start_time = time.time()
        for is_finished in self._async_copies:
            timeout = max([0.0, self._MAX_COPY_SECONDS - (time.time() - start_time)])
            if not is_finished.wait(timeout):
                raise ValueError('Timed out waiting for copy to complete.')
        if self._errors:
            raise self._errors[0]