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
def _as_shape_list(shapes, dtypes, unknown_dim_allowed=False, unknown_rank_allowed=False):
    """Convert shapes to a list of tuples of int (or None)."""
    del dtypes
    if unknown_dim_allowed:
        if not isinstance(shapes, collections_abc.Sequence) or not shapes or any((shape is None or isinstance(shape, int) for shape in shapes)):
            raise ValueError('When providing partial shapes, a list of shapes must be provided.')
    if shapes is None:
        return None
    if isinstance(shapes, tensor_shape.TensorShape):
        shapes = [shapes]
    if not isinstance(shapes, (tuple, list)):
        raise TypeError(f'Shapes must be a TensorShape or a list or tuple of TensorShapes, got {type(shapes)} instead.')
    if all((shape is None or isinstance(shape, int) for shape in shapes)):
        shapes = [shapes]
    shapes = [tensor_shape.as_shape(shape) for shape in shapes]
    if not unknown_dim_allowed:
        if any((not shape.is_fully_defined() for shape in shapes)):
            raise ValueError(f'All shapes must be fully defined: {shapes}')
    if not unknown_rank_allowed:
        if any((shape.dims is None for shape in shapes)):
            raise ValueError(f'All shapes must have a defined rank: {shapes}')
    return shapes