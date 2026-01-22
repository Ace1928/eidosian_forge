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
class MetricsAggregator(Aggregator):
    """Aggregator that calculates loss and metrics info.

  Attributes:
    use_steps: Whether the loop is using `step` or `batch_size`.
    num_samples: Total number of samples: `batch_size*num_batches`.
    steps: Total number of steps, ie number of times to iterate over a dataset
      to cover all samples.
  """

    def __init__(self, use_steps, num_samples=None, steps=None):
        super(MetricsAggregator, self).__init__(use_steps=use_steps, num_samples=num_samples, steps=steps, batch_size=None)

    def create(self, batch_outs):
        self.results = [0.0] * len(batch_outs)

    def aggregate(self, batch_outs, batch_start=None, batch_end=None):
        if self.use_steps:
            self.results[0] += batch_outs[0]
        else:
            self.results[0] += batch_outs[0] * (batch_end - batch_start)
        self.results[1:] = batch_outs[1:]

    def finalize(self):
        if not self.results:
            raise ValueError('Empty training data.')
        self.results[0] /= self.num_samples or self.steps