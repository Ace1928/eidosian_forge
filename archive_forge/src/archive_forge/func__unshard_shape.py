from tensorflow.python.framework import tensor_shape
def _unshard_shape(self, shape):
    """Return the unsharded shape that would generate a given sharded shape.

    Args:
      shape: the sharded shape to unshard

    Returns:
      The unsharded shape.

    Raises:
      ValueError: if shape is unknown or does not contain
        self.shard_dimension
      TypeError: if shape is not convertible to a TensorShape
    """
    shape = tensor_shape.as_shape(shape)
    if self._number_of_shards == 1:
        return shape
    ndims = shape.ndims
    if ndims is None:
        raise ValueError(f'Shape {shape} must be statically known.')
    if ndims <= self._shard_dimension:
        raise ValueError(f'Shape {shape.as_list()} does not contain shard_dimension {self._shard_dimension}. Rank is too small.')
    dims = shape.as_list()
    dims[self._shard_dimension] *= self._number_of_shards
    return tensor_shape.TensorShape(dims)