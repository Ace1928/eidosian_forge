from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def actual_grad_fn(*result_grad_components):
    """Custom grad fn wrapper."""
    result_grads = composite_tensor_gradient.replace_flat_tensors_for_gradients(nest.flatten(result), result_grad_components)
    if not isinstance(result_grads, (list, tuple)):
        result_grads = [result_grads]
    if variables:
        input_grads, variable_grads = grad_fn(*result_grads, variables=variables)
        if len(variable_grads) != len(variables):
            raise ValueError('Must return gradient for each variable from @custom_gradient grad_fn.')
    else:
        input_grads = grad_fn(*result_grads)
        variable_grads = []
    flat_grads = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(input_grads))
    if len(flat_grads) != arg_count:
        raise ValueError(f'custom_gradient function expected to return {arg_count} gradients, but returned {len(flat_grads)} instead.')
    return flat_grads + variable_grads