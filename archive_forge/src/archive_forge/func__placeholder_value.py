import functools
import sys
import traceback
import numpy as np
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import variables
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import misc
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.types import distribute
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
def _placeholder_value(like, shape_invariant, original=None):
    """Constructs a (dummy) placeholder value for a loop-initialized variable.

  Args:
    like: Any object. The value created by the first iteration of the loop. If a
      Python scalar, the placeholder will be the zero value of that type. If a
      Tensor, the placeholder will be a zero tensor of matching shape and dtype.
      If a list, dict or tuple, the placeholder will be an identical structure
      of placeholders.
    shape_invariant: The shape invariant specified by the user (or None, if
      nothing was specified) for the respective variable.
    original: Any object. The value of the variable prior to entering the loop.
      Typically, this is one of the special "Undefined" value, because that's
      when a placeholder is needed.

  Returns:
    Either a zero value of structure, shape and dtype mathing 'like', or
    'original', if no such zero value could be created.
  """
    if like is None:
        return (original, None)
    elif isinstance(like, (variables.Undefined, variables.UndefinedReturnValue)):
        return (original, None)
    elif isinstance(like, (int, float, bool)):
        return (type(like)(0), None)
    elif tensor_util.is_tf_type(like):
        like_shape = shape_invariant if shape_invariant is not None else like.shape
        if like_shape is None or like_shape.rank is None:
            return (array_ops.zeros((), like.dtype), like_shape)
        placeholder_shape = []
        has_dynamic_dims = False
        for s, i in zip(like.shape, like_shape):
            if i is None:
                like_dim = 0
            elif isinstance(i, tensor_shape.Dimension):
                if i.value is None:
                    like_dim = 0
                else:
                    like_dim = i.value
            else:
                like_dim = i
            if s is None:
                placeholder_shape.append(like_dim)
                has_dynamic_dims = True
            elif isinstance(s, tensor_shape.Dimension):
                if s.value is None:
                    placeholder_shape.append(like_dim)
                    has_dynamic_dims = True
                else:
                    placeholder_shape.append(s.value)
            else:
                placeholder_shape.append(s)
        if has_dynamic_dims:
            invariant = like_shape
        else:
            invariant = None
        return (array_ops.zeros(placeholder_shape, like.dtype), invariant)
    elif isinstance(like, (list, tuple, dict)):
        if shape_invariant is None:
            zipped = nest.map_structure(lambda v: _placeholder_value(v, None), nest.flatten(like))
        else:
            zipped = nest.map_structure(_placeholder_value, nest.flatten(like), nest.flatten(shape_invariant))
        vals, invars = zip(*zipped)
        return (nest.pack_sequence_as(like, vals), nest.pack_sequence_as(like, invars))
    raise TypeError("Found an unsupported type '{}' while creating placeholder for {}. Supported types include Tensor, int, float, bool, list, tuple or dict.".format(type(like).__name__, like))