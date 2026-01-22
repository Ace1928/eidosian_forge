from absl import logging
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import attributes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.trackable import base
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
def _get_signature_name_changes(concrete_function):
    """Checks for user-specified signature input names that are normalized."""
    name_changes = {}
    for signature_input_name, graph_input in zip(concrete_function.function_def.signature.input_arg, concrete_function.graph.inputs):
        try:
            user_specified_name = compat.as_str(graph_input.op.get_attr('_user_specified_name'))
            if signature_input_name.name != user_specified_name:
                name_changes[user_specified_name] = signature_input_name.name
        except ValueError:
            pass
    return name_changes