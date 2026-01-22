from typing import Optional
import inspect
import sys
import warnings
from functools import wraps
def _append_doc(obj, *, message: str, directive: Optional[str]=None) -> str:
    if not obj.__doc__:
        obj.__doc__ = ''
    obj.__doc__ = obj.__doc__.rstrip()
    indent = _get_indent(obj.__doc__)
    obj.__doc__ += '\n\n'
    if directive is not None:
        obj.__doc__ += f'{' ' * indent}.. {directive}::\n\n'
        message = message.replace('\n', '\n' + ' ' * (indent + 4))
        obj.__doc__ += f'{' ' * (indent + 4)}{message}'
    else:
        message = message.replace('\n', '\n' + ' ' * (indent + 4))
        obj.__doc__ += f'{' ' * indent}{message}'
    obj.__doc__ += f'\n{' ' * indent}'