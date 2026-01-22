import functools
import inspect
from typing import Any, Dict, Tuple
import six
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import nest
def _validate_signature(signature):
    """Checks the input_signature to be valid."""
    if signature is None:
        return
    if not isinstance(signature, (tuple, list)):
        raise TypeError(f'input_signature must be either a tuple or a list, got {type(signature)}.')
    variable_specs = _get_variable_specs(signature)
    if variable_specs:
        raise TypeError(f"input_signature doesn't support VariableSpec, got {variable_specs}")
    if any((not isinstance(arg, tensor.TensorSpec) for arg in nest.flatten(signature, expand_composites=True))):
        bad_args = [arg for arg in nest.flatten(signature, expand_composites=True) if not isinstance(arg, tensor.TensorSpec)]
        raise TypeError(f'input_signature must be a possibly nested sequence of TensorSpec objects, got invalid args {bad_args} with types {list(six.moves.map(type, bad_args))}.')