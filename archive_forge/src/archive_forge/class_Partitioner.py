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
@tf_export('distribute.experimental.partitioners.Partitioner', v1=[])
class Partitioner(object):
    """Partitioner base class: all partitiners inherit from this class.

  Partitioners should implement a `__call__` method with the following
  signature:

  ```python
  def __call__(self, shape, dtype, axis=0):
    # Partitions the given `shape` and returns the partition results.
    # See docstring of `__call__` method for the format of partition results.
  ```
  """

    def __call__(self, shape, dtype, axis=0):
        """Partitions the given `shape` and returns the partition results.

    Examples of a partitioner that allocates a fixed number of shards:

    ```python
    partitioner = FixedShardsPartitioner(num_shards=2)
    partitions = partitioner(tf.TensorShape([10, 3], tf.float32), axis=0)
    print(partitions) # [2, 0]
    ```

    Args:
      shape: a `tf.TensorShape`, the shape to partition.
      dtype: a `tf.dtypes.Dtype` indicating the type of the partition value.
      axis: The axis to partition along.  Default: outermost axis.

    Returns:
      A list of integers representing the number of partitions on each axis,
      where i-th value correponds to i-th axis.
    """
        raise NotImplementedError