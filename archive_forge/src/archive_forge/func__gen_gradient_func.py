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
def _gen_gradient_func(func):
    """Wraps a deserialized function."""

    def gradient_func(unused_op, *result_grads):

        def none_to_zero(x, t):
            if x is not None:
                return x
            shape, dtype = default_gradient.shape_and_dtype(t)
            if shape.is_fully_defined():
                return default_gradient.zeros_like(t)
            dims = []
            if shape.rank is not None:
                dims = [1 if d is None else d for d in shape.as_list()]
            return array_ops.zeros(dims, dtype)
        result_grads = [none_to_zero(x, t) for x, t in zip(result_grads, func.graph.inputs)]
        return func(*result_grads)
    return gradient_func