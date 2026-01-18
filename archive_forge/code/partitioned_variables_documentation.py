import math
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
Partitioner that partitions list for a variable of given shape and type.

    Ex: Consider partitioning a variable of type float32 with
      shape=[1024, 1024].
      If `max_partitions` >= 16, this function would return
        [(1024 * 1024 * 4) / (256 * 1024), 1] = [16, 1].
      If `max_partitions` < 16, this function would return
        [`max_partitions`, 1].

    Args:
      shape: Shape of the variable.
      dtype: Type of the variable.

    Returns:
      List of partitions for each axis (currently only one axis can be
      partitioned).

    Raises:
      ValueError: If axis to partition along does not exist for the variable.
    