import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class RecordInput:
    """RecordInput asynchronously reads and randomly yields TFRecords.

  A RecordInput Op will continuously read a batch of records asynchronously
  into a buffer of some fixed capacity. It can also asynchronously yield
  random records from this buffer.

  It will not start yielding until at least `buffer_size / 2` elements have been
  placed into the buffer so that sufficient randomization can take place.

  The order the files are read will be shifted each epoch by `shift_amount` so
  that the data is presented in a different order every epoch.
  """

    def __init__(self, file_pattern, batch_size=1, buffer_size=1, parallelism=1, shift_ratio=0, seed=0, name=None, batches=None, compression_type=None):
        """Constructs a RecordInput Op.

    Args:
      file_pattern: File path to the dataset, possibly containing wildcards.
        All matching files will be iterated over each epoch.
      batch_size: How many records to return at a time.
      buffer_size: The maximum number of records the buffer will contain.
      parallelism: How many reader threads to use for reading from files.
      shift_ratio: What percentage of the total number files to move the start
        file forward by each epoch.
      seed: Specify the random number seed used by generator that randomizes
        records.
      name: Optional name for the operation.
      batches: None by default, creating a single batch op. Otherwise specifies
        how many batches to create, which are returned as a list when
        `get_yield_op()` is called. An example use case is to split processing
        between devices on one computer.
      compression_type: The type of compression for the file. Currently ZLIB and
        GZIP are supported. Defaults to none.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
        self._batch_size = batch_size
        if batches is not None:
            self._batch_size *= batches
        self._batches = batches
        self._file_pattern = file_pattern
        self._buffer_size = buffer_size
        self._parallelism = parallelism
        self._shift_ratio = shift_ratio
        self._seed = seed
        self._name = name
        self._compression_type = python_io.TFRecordCompressionType.NONE
        if compression_type is not None:
            self._compression_type = compression_type

    def get_yield_op(self):
        """Adds a node that yields a group of records every time it is executed.
    If RecordInput `batches` parameter is not None, it yields a list of
    record batches with the specified `batch_size`.
    """
        compression_type = python_io.TFRecordOptions.get_compression_type_string(python_io.TFRecordOptions(self._compression_type))
        records = gen_data_flow_ops.record_input(file_pattern=self._file_pattern, file_buffer_size=self._buffer_size, file_parallelism=self._parallelism, file_shuffle_shift_ratio=self._shift_ratio, batch_size=self._batch_size, file_random_seed=self._seed, compression_type=compression_type, name=self._name)
        if self._batches is None:
            return records
        else:
            with ops.name_scope(self._name):
                batch_list = [[] for _ in range(self._batches)]
                records = array_ops.split(records, self._batch_size, 0)
                for index, protobuf in enumerate(records):
                    batch_index = index % self._batches
                    batch_list[batch_index].append(array_ops.reshape(protobuf, []))
                return batch_list