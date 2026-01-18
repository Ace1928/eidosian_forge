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
def to_fullargspec(function_type: function_type_lib.FunctionType, default_values: Dict[str, Any]) -> inspect.FullArgSpec:
    """Generates backwards compatible FullArgSpec from FunctionType."""
    args = []
    varargs = None
    varkw = None
    defaults = []
    kwonlyargs = []
    kwonlydefaults = {}
    for parameter in function_type.parameters.values():
        if parameter.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]:
            args.append(parameter.name)
            if parameter.default is not inspect.Parameter.empty:
                defaults.append(default_values[parameter.name])
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwonlyargs.append(parameter.name)
            if parameter.default is not inspect.Parameter.empty:
                kwonlydefaults[parameter.name] = default_values[parameter.name]
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            varargs = parameter.name
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            varkw = parameter.name
    return inspect.FullArgSpec(args, varargs, varkw, tuple(defaults) if defaults else None, kwonlyargs, kwonlydefaults if kwonlydefaults else None, annotations={})