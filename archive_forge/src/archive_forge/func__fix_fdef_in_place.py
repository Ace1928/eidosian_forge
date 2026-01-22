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
def _fix_fdef_in_place(fdef, functions, shared_name_suffix, new_gradient_op_types):
    """Fixes a FunctionDef proto to be loaded in current context.

  In particular, when loading a function library into an eager context, one
  must rename the functions to avoid conflicts with existent functions.

  Args:
    fdef: FunctionDef proto to fix. It is mutated in-place.
    functions: map from function name to a ConcreteFunction instance.
    shared_name_suffix: A unique string for this load which helps to avoid
      `shared_name` collisions across loads. Two functions from the same load
      using the same `shared_name` still need to share, but functions from
      different loads with the same `shared_name` should not.
    new_gradient_op_types: map from old gradient op type to newly generated op
      type.

  Returns:
    orig_name: original value of fdef.signature.name
  """
    orig_name = fdef.signature.name
    contains_unsaved_custom_gradients = False
    for node_def in fdef.node_def:
        fix_node_def(node_def, functions, shared_name_suffix)
        op_type = _get_gradient_op_type(node_def)
        if op_type is not None:
            if op_type in new_gradient_op_types:
                node_def.attr['_gradient_op_type'].s = compat.as_bytes(new_gradient_op_types[op_type])
            else:
                contains_unsaved_custom_gradients = True
    if contains_unsaved_custom_gradients:
        logging.warning('Importing a function (%s) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.', fdef.signature.name)
    fdef.signature.name = _clean_function_name(fdef.signature.name)
    return orig_name