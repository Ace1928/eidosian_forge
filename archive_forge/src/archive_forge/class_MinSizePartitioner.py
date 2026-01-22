import copy
import math
from typing import Sequence
import weakref
import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('distribute.experimental.partitioners.MinSizePartitioner', v1=[])
class MinSizePartitioner(Partitioner):
    """Partitioner that allocates a minimum size per shard.

  This partitioner ensures each shard has at least `min_shard_bytes`, and tries
  to allocate as many shards as possible, i.e., keeping shard size as small as
  possible. The maximum number of such shards (upper bound) is given by
  `max_shards`.

  Examples:

  >>> partitioner = MinSizePartitioner(min_shard_bytes=4, max_shards=2)
  >>> partitions = partitioner(tf.TensorShape([6, 1]), tf.float32)
  >>> [2, 1]
  >>> partitioner = MinSizePartitioner(min_shard_bytes=4, max_shards=10)
  >>> partitions = partitioner(tf.TensorShape([6, 1]), tf.float32)
  >>> [6, 1]
  >>>
  >>> # use in ParameterServerStrategy
  >>> # strategy = tf.distribute.experimental.ParameterServerStrategy(
  >>> #   cluster_resolver=cluster_resolver, variable_partitioner=partitioner)
  """

    def __init__(self, min_shard_bytes=256 << 10, max_shards=1, bytes_per_string=16):
        """Creates a new `MinSizePartitioner`.

    Args:
      min_shard_bytes: Minimum bytes of each shard. Defaults to 256K.
      max_shards: Upper bound on the number of shards. Defaults to 1.
      bytes_per_string: If the partition value is of type string, this provides
        an estimate of how large each string is.
    """
        if min_shard_bytes < 1:
            raise ValueError(f'Argument `min_shard_bytes` must be positive. Received: {min_shard_bytes}')
        if max_shards < 1:
            raise ValueError(f'Argument `max_shards` must be positive. Received: {max_shards}')
        if bytes_per_string < 1:
            raise ValueError(f'Argument `bytes_per_string` must be positive. Received: {bytes_per_string}')
        self._min_shard_bytes = min_shard_bytes
        self._max_shards = max_shards
        self._bytes_per_string = bytes_per_string

    def __call__(self, shape, dtype, axis=0):
        return partitioned_variables.min_max_variable_partitioner(max_partitions=self._max_shards, axis=axis, min_slice_size=self._min_shard_bytes, bytes_per_string_element=self._bytes_per_string)(shape, dtype)