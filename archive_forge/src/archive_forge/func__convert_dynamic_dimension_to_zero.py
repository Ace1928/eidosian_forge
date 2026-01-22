import collections
from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def _convert_dynamic_dimension_to_zero(shape):
    """Converts dynamic dimensions in `shape` to zero.

  The fake params created to match the intermediates captured in other branches
  could have dynamic dimensions. But the XLA shape is not able to handle
  dynamic dimensions in TF TensorShape. Setting the dynamic dimensions to
  size zero will help avoid failing safety checks in bridge. When XLA
  DynamicConditional op reconciles branch differences, XLA will replace the
  dimension size 0 with a bounded dimension determined from the shape of
  real argument in the other branch.

  Note: Rank unknown shapes are returned as they are.

  Args:
    shape: The TensorShape of fake param.

  Returns:
    The new TensorShape with dynamic dimensions set to zero.
  """
    if shape.rank is None:
        return shape
    return tensor_shape.TensorShape([0 if d is None else d for d in shape.as_list()])