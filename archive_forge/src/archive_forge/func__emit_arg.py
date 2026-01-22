import inspect
import textwrap
import torch.jit
from torch.jit._builtins import _find_builtin
def _emit_arg(indent, i, arg):
    v = f'{arg.name} : {_emit_type(arg.type)}'
    default = arg.default_value
    if default is not None:
        v = f'{v}={str(default)}'
    if i > 0:
        v = f'\n{' ' * indent}{v}'
    return v