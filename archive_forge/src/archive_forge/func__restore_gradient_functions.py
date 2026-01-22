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
def _restore_gradient_functions(func_graph, renamed_functions, loaded_gradients):
    """Populate function op's _gradient_function with default gradient."""
    for op in func_graph.get_operations():
        if op.type in ['StatefulPartitionedCall', 'PartitionedCall']:
            function = renamed_functions[compat.as_bytes(op.node_def.attr['f'].func.name)]
            op._gradient_function = function._get_gradient_function()
        try:
            gradient_op_type = op.get_attr('_gradient_op_type')
        except ValueError:
            pass
        else:
            if gradient_op_type in loaded_gradients:
                grad_fn = loaded_gradients[gradient_op_type]
                grad_fn._num_positional_args = len(op.inputs)
                grad_fn._arg_keywords = [inp.name for inp in op.inputs]