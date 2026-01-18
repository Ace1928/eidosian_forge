import itertools
import numpy as np
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_sharding
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
Checks dims and partitions or replicates the input tensor.

      The ops inside this function are placed on the host side.

    Args:
      tensor: The input tensor which will be partitioned or replicated.
      dims: A list of integer describes how to partition the input tensor.

    Returns:
      An iterator of `Tensor`s or a list of partitioned tensors.
    