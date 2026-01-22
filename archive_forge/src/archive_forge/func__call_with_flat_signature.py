import collections
import pprint
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import record
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
def _call_with_flat_signature(self, args, kwargs):
    """Executes the wrapped function with the flat signature.

    Args:
      args: Positional arguments to the concrete function.
      kwargs: Keyword arguments to the concrete function.

    Returns:
      The result of applying the function on the Tensors/Variables contained in
      `args` and `kwargs`.
    Raises:
      TypeError: if `args` and `kwargs` do not match the flat signature of this
        `ConcreteFunction`.
    """
    if len(args) > self._num_positional_args:
        raise TypeError(f'{self._flat_signature_summary()} takes {self._num_positional_args} positional arguments, got {len(args)}.')
    args = list(args)
    kwargs = dict(kwargs)
    kwargs = {function_type_lib.sanitize_arg_name(k): v for k, v in kwargs.items()}
    for keyword in self._arg_keywords[len(args):]:
        try:
            args.append(kwargs.pop(function_type_lib.sanitize_arg_name(compat.as_str(keyword))))
        except KeyError:
            specified_keywords = list(self._arg_keywords[:len(args)]) + list(kwargs.keys())
            missing_required_args = sorted(set(self._arg_keywords) - set(specified_keywords))
            raise TypeError(f'{self._flat_signature_summary()} missing required arguments: {', '.join(missing_required_args)}.')
    if kwargs:
        positional_arg_keywords = set(self._arg_keywords[:len(args)])
        for unused_key in kwargs:
            if unused_key in positional_arg_keywords:
                raise TypeError(f"{self._flat_signature_summary()} got two values for '{unused_key}'.")
        raise TypeError(f'{self._flat_signature_summary()} got unexpected keyword arguments: {', '.join(sorted(kwargs))}.')
    for i, arg in enumerate(args):
        if not isinstance(arg, (tensor_lib.Tensor, resource_variable_ops.BaseResourceVariable)):
            raise TypeError(f'{self._flat_signature_summary()}: expected argument #{i}(zero-based) to be a Tensor; got {type(arg).__name__} ({arg}).')
    return self._call_flat(args, self.captured_inputs)