import functools
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.parallel_for.pfor import PFor
from tensorflow.python.ops.parallel_for.pfor import PForConfig
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _composite_to_tensors(value, is_batched=False):
    """Converts a CompositeTensor into a list of stackable tensors."""
    if _should_expand_composite(value):
        spec = value._type_spec
        if not isinstance(spec, type_spec.BatchableTypeSpec):
            raise ValueError(f'CompositeTensor instance {value} returned from parallel_for or vectorized_map loop body must provide a `BatchableTypeSpec` (saw: {spec}).')
        if is_batched:
            return spec._to_batched_tensor_list(value)
        return spec._to_tensor_list(value)
    return value