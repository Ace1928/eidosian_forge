import collections
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util_v1
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2_indexed_slices_rewriter
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
def _get_structured_grad_output(outputs, grads, body_grad_graph):
    """Returns the values that should be returned from the while grad function.

  Args:
    outputs: the raw Tensor outputs of the grad While op.
    grads: the input gradients to the gradient function.
    body_grad_graph: _WhileBodyGradFuncGraph.

  Returns:
    A list of gradient values. May include Nones.
  """
    result = []
    outputs_idx = 3
    structured_outputs_idx = 3
    for g in grads:
        if g is None:
            result.append(None)
            continue
        output = body_grad_graph.structured_outputs[structured_outputs_idx]
        structured_outputs_idx += 1
        if isinstance(output, indexed_slices.IndexedSlices):
            result.append(indexed_slices.IndexedSlices(values=outputs[outputs_idx], indices=outputs[outputs_idx + 1], dense_shape=outputs[outputs_idx + 2]))
            outputs_idx += 3
        else:
            assert isinstance(output, tensor_lib.Tensor)
            result.append(outputs[outputs_idx])
            outputs_idx += 1
    return result