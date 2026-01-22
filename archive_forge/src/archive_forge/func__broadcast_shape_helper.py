import itertools
from tensorflow.python.framework import tensor_shape
def _broadcast_shape_helper(shape_x, shape_y):
    """Helper functions for is_broadcast_compatible and broadcast_shape.

  Args:
    shape_x: A `TensorShape`
    shape_y: A `TensorShape`

  Returns:
    Returns None if the shapes are not broadcast compatible,
    a list of the broadcast dimensions otherwise.
  """
    broadcasted_dims = reversed(list(itertools.zip_longest(reversed(shape_x.dims), reversed(shape_y.dims), fillvalue=tensor_shape.Dimension(1))))
    return_dims = []
    for dim_x, dim_y in broadcasted_dims:
        if dim_x.value is None or dim_y.value is None:
            if dim_x.value is not None and dim_x.value > 1:
                return_dims.append(dim_x)
            elif dim_y.value is not None and dim_y.value > 1:
                return_dims.append(dim_y)
            else:
                return_dims.append(None)
        elif dim_x.value == 1:
            return_dims.append(dim_y)
        elif dim_y.value == 1:
            return_dims.append(dim_x)
        elif dim_x.value == dim_y.value:
            return_dims.append(dim_x.merge_with(dim_y))
        else:
            return None
    return return_dims