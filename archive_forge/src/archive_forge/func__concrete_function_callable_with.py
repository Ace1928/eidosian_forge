import collections
import pprint
import re
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_lib
from tensorflow.python.framework import function_def_to_graph as function_def_lib
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def _concrete_function_callable_with(function, inputs, allow_conversion):
    """Returns whether concrete `function` can be called with `inputs`."""
    expected_structure = function.graph.structured_input_signature
    try:
        flatten_inputs = nest.flatten_up_to(expected_structure, inputs)
    except (TypeError, ValueError):
        return False
    for arg, expected in zip(flatten_inputs, nest.flatten(expected_structure)):
        if isinstance(expected, tensor.TensorSpec):
            if allow_conversion:
                arg = _try_convert_to_tensor_spec(arg, dtype_hint=expected.dtype)
            if not _is_tensor(arg) and (not isinstance(arg, tensor.TensorSpec)):
                return False
            if arg.dtype != expected.dtype:
                return False
            if not expected.shape.is_compatible_with(arg.shape):
                return False
        elif isinstance(expected, type_spec.TypeSpec):
            if not expected.is_compatible_with(arg):
                return False
        elif _is_tensor(arg):
            if id(arg) != id(expected):
                return False
        elif arg != expected:
            return False
    return True