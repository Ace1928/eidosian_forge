import math
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['variable_axis_size_partitioner'])
def variable_axis_size_partitioner(max_shard_bytes, axis=0, bytes_per_string_element=16, max_shards=None):
    """Get a partitioner for VariableScope to keep shards below `max_shard_bytes`.

  This partitioner will shard a Variable along one axis, attempting to keep
  the maximum shard size below `max_shard_bytes`.  In practice, this is not
  always possible when sharding along only one axis.  When this happens,
  this axis is sharded as much as possible (i.e., every dimension becomes
  a separate shard).

  If the partitioner hits the `max_shards` limit, then each shard may end up
  larger than `max_shard_bytes`. By default `max_shards` equals `None` and no
  limit on the number of shards is enforced.

  One reasonable value for `max_shard_bytes` is `(64 << 20) - 1`, or almost
  `64MB`, to keep below the protobuf byte limit.

  Args:
    max_shard_bytes: The maximum size any given shard is allowed to be.
    axis: The axis to partition along.  Default: outermost axis.
    bytes_per_string_element: If the `Variable` is of type string, this provides
      an estimate of how large each scalar in the `Variable` is.
    max_shards: The maximum number of shards in int created taking precedence
      over `max_shard_bytes`.

  Returns:
    A partition function usable as the `partitioner` argument to
    `variable_scope` and `get_variable`.

  Raises:
    ValueError: If any of the byte counts are non-positive.
  """
    if max_shard_bytes < 1 or bytes_per_string_element < 1:
        raise ValueError(f'Both max_shard_bytes and bytes_per_string_element must be positive. Currently, max_shard_bytes is {max_shard_bytes} andbytes_per_string_element is {bytes_per_string_element}')
    if max_shards and max_shards < 1:
        raise ValueError('max_shards must be positive.')

    def _partitioner(shape, dtype):
        """Partitioner that partitions shards to have max_shard_bytes total size.

    Args:
      shape: A `TensorShape`.
      dtype: A `DType`.

    Returns:
      A tuple representing how much to slice each axis in shape.

    Raises:
      ValueError: If shape is not a fully defined `TensorShape` or dtype is not
        a `DType`.
    """
        if not isinstance(shape, tensor_shape.TensorShape):
            raise ValueError(f'shape is not a TensorShape: {shape}')
        if not shape.is_fully_defined():
            raise ValueError(f'shape is not fully defined: {shape}')
        if not isinstance(dtype, dtypes.DType):
            raise ValueError(f'dtype is not a DType: {dtype}')
        if dtype.base_dtype == dtypes.string:
            element_size = bytes_per_string_element
        else:
            element_size = dtype.size
        partitions = [1] * shape.ndims
        bytes_per_slice = 1.0 * (shape.num_elements() / shape.dims[axis].value) * element_size
        slices_per_shard = max(1, math.floor(max_shard_bytes / bytes_per_slice))
        axis_shards = int(math.ceil(1.0 * shape.dims[axis].value / slices_per_shard))
        if max_shards:
            axis_shards = min(max_shards, axis_shards)
        partitions[axis] = axis_shards
        return partitions
    return _partitioner