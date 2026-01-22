import functools
import sys
import time
import six
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.distribute_lib import InputReplicationMode
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import distribute as distribute_types
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
def _get_batched_dataset_attributes(d):
    """Get `batch_size`, `drop_remainder` of dataset."""
    assert isinstance(d, (dataset_ops.BatchDataset, batching._MapAndBatchDataset))
    if isinstance(d, dataset_ops.BatchDataset):
        batch_size = d._batch_size
        drop_remainder = d._drop_remainder
    elif isinstance(d, batching._MapAndBatchDataset):
        batch_size = d._batch_size_t
        drop_remainder = d._drop_remainder_t
    if tensor_util.is_tf_type(batch_size):
        batch_size = tensor_util.constant_value(batch_size)
    if tensor_util.is_tf_type(drop_remainder):
        drop_remainder = tensor_util.constant_value(drop_remainder)
    return (batch_size, drop_remainder)