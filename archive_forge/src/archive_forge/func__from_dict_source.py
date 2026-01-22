import sys
import types
import typing
import warnings
import simdjson as json
from enum import Enum
from dataclasses import is_dataclass
from .utils import issubclass_safe
def _from_dict_source(cls):
    lines = ['def from_dict(cls, o):', '    args = []']
    for name, field_type in typing.get_type_hints(cls).items():
        if typing.get_origin(field_type) == typing.Union:
            field_type = typing.get_args(field_type)[0]
        access = f'o.get({name!r})'
        transform = expr_builder_from(field_type)
        if transform('x') != 'x':
            lines.append(f'    value = {access}')
            lines.append(f'    if value is not None:')
            lines.append(f'        value = ' + transform('value'))
            lines.append(f'    args.append(value)')
        else:
            lines.append(f'    args.append({access})')
    lines.append('    return cls(*args)')
    lines.append('')
    return '\n'.join(lines)