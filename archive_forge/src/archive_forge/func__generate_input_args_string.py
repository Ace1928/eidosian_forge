import inspect
import functools
from enum import Enum
import torch.autograd
def _generate_input_args_string(obj):
    """Generate a string for the input arguments of an object."""
    signature = inspect.signature(obj.__class__)
    input_param_names = set()
    for param_name in signature.parameters.keys():
        input_param_names.add(param_name)
    result = []
    for name, value in inspect.getmembers(obj):
        if name in input_param_names:
            result.append((name, _simplify_obj_name(value)))
    return ', '.join([f'{name}={value}' for name, value in result])