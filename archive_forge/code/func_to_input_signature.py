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
def to_input_signature(function_type):
    """Extracts an input_signature from function_type instance."""
    constrained_parameters = list(function_type.parameters.keys())
    if 'self' in constrained_parameters:
        constrained_parameters.pop(0)
    if not constrained_parameters:
        return tuple()
    constraints = []
    is_auto_constrained = False
    for parameter_name in constrained_parameters:
        parameter = function_type.parameters[parameter_name]
        constraint = None
        if parameter.type_constraint:
            constraint = parameter.type_constraint.placeholder_value(trace_type.InternalPlaceholderContext(unnest_only=True))
            if any((not isinstance(arg, tensor.TensorSpec) for arg in nest.flatten([constraint], expand_composites=True))):
                is_auto_constrained = True
                break
            else:
                constraints.append(constraint)
    if is_auto_constrained and (not constraints):
        return tuple()
    return tuple(constraints) if constraints else None