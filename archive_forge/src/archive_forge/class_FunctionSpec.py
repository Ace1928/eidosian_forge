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
class FunctionSpec(object):
    """Specification of how to bind arguments to a function.

  Deprecated. Please use FunctionType instead.
  """

    @classmethod
    def from_function_and_signature(cls, python_function, input_signature, is_pure=False, jit_compile=None):
        """Creates a FunctionSpec instance given a python function and signature.

    Args:
      python_function: a function to inspect
      input_signature: a signature of the function (None, if variable)
      is_pure: if True all input arguments (including variables and constants)
        will be converted to tensors and no variable changes allowed.
      jit_compile: see `tf.function`

    Returns:
      instance of FunctionSpec
    """
        function_type, default_values = make_function_type(python_function, input_signature)
        while isinstance(python_function, functools.partial):
            python_function = python_function.func
        name = getattr(python_function, '__name__', 'f')
        return FunctionSpec(function_type, default_values, is_pure=is_pure, jit_compile=jit_compile, name=name)

    @classmethod
    def from_fullargspec_and_signature(cls, fullargspec, input_signature, is_pure=False, name=None, jit_compile=None):
        """Construct FunctionSpec from legacy FullArgSpec format."""
        function_type, default_values = to_function_type(fullargspec)
        if input_signature:
            input_signature = tuple(input_signature)
            _validate_signature(input_signature)
            function_type = function_type_lib.add_type_constraints(function_type, input_signature, default_values)
        return FunctionSpec(function_type, default_values, is_pure, name, jit_compile)

    def __init__(self, function_type, default_values, is_pure=False, name=None, jit_compile=None):
        """Constructs a FunctionSpec describing a python function.

    Args:
      function_type: A FunctionType describing the python function signature.
      default_values: Dictionary mapping parameter names to default values.
      is_pure: if True all input arguments (including variables and constants)
        will be converted to tensors and no variable changes allowed.
      name: Name of the function
      jit_compile: see `tf.function`.
    """
        self._function_type = function_type
        self._default_values = default_values
        self._fullargspec = to_fullargspec(function_type, default_values)
        self._is_pure = is_pure
        self._jit_compile = jit_compile
        self._name = name or 'f'
        self._input_signature = to_input_signature(function_type)

    @property
    def default_values(self):
        """Returns dict mapping parameter names to default values."""
        return self._default_values

    @property
    def function_type(self):
        """Returns a FunctionType representing the Python function signature."""
        return self._function_type

    @property
    def fullargspec(self):
        return self._fullargspec

    @property
    def input_signature(self):
        return self._input_signature

    @property
    def flat_input_signature(self):
        return tuple(nest.flatten(self.input_signature, expand_composites=True))

    @property
    def is_pure(self):
        return self._is_pure

    @property
    def jit_compile(self):
        return self._jit_compile

    @property
    def arg_names(self):
        return to_arg_names(self.function_type)

    def signature_summary(self, default_values=False):
        """Returns a string summarizing this function's signature.

    Args:
      default_values: If true, then include default values in the signature.

    Returns:
      A `string`.
    """
        summary = f'{self._function_type!r}'
        if default_values:
            summary += f', defaults: {self.default_values!r}'
        return summary