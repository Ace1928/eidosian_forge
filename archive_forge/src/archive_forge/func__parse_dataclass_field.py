import dataclasses
import json
import sys
import types
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError
from copy import copy
from enum import Enum
from inspect import isclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, NewType, Optional, Tuple, Union, get_type_hints
import yaml
@staticmethod
def _parse_dataclass_field(parser: ArgumentParser, field: dataclasses.Field):
    field_name = f'--{field.name}'
    kwargs = field.metadata.copy()
    if isinstance(field.type, str):
        raise RuntimeError('Unresolved type detected, which should have been done with the help of `typing.get_type_hints` method by default')
    aliases = kwargs.pop('aliases', [])
    if isinstance(aliases, str):
        aliases = [aliases]
    origin_type = getattr(field.type, '__origin__', field.type)
    if origin_type is Union or (hasattr(types, 'UnionType') and isinstance(origin_type, types.UnionType)):
        if str not in field.type.__args__ and (len(field.type.__args__) != 2 or type(None) not in field.type.__args__):
            raise ValueError(f"Only `Union[X, NoneType]` (i.e., `Optional[X]`) is allowed for `Union` because the argument parser only supports one type per argument. Problem encountered in field '{field.name}'.")
        if type(None) not in field.type.__args__:
            field.type = field.type.__args__[0] if field.type.__args__[1] == str else field.type.__args__[1]
            origin_type = getattr(field.type, '__origin__', field.type)
        elif bool not in field.type.__args__:
            field.type = field.type.__args__[0] if isinstance(None, field.type.__args__[1]) else field.type.__args__[1]
            origin_type = getattr(field.type, '__origin__', field.type)
    bool_kwargs = {}
    if origin_type is Literal or (isinstance(field.type, type) and issubclass(field.type, Enum)):
        if origin_type is Literal:
            kwargs['choices'] = field.type.__args__
        else:
            kwargs['choices'] = [x.value for x in field.type]
        kwargs['type'] = make_choice_type_function(kwargs['choices'])
        if field.default is not dataclasses.MISSING:
            kwargs['default'] = field.default
        else:
            kwargs['required'] = True
    elif field.type is bool or field.type == Optional[bool]:
        bool_kwargs = copy(kwargs)
        kwargs['type'] = string_to_bool
        if field.type is bool or (field.default is not None and field.default is not dataclasses.MISSING):
            default = False if field.default is dataclasses.MISSING else field.default
            kwargs['default'] = default
            kwargs['nargs'] = '?'
            kwargs['const'] = True
    elif isclass(origin_type) and issubclass(origin_type, list):
        kwargs['type'] = field.type.__args__[0]
        kwargs['nargs'] = '+'
        if field.default_factory is not dataclasses.MISSING:
            kwargs['default'] = field.default_factory()
        elif field.default is dataclasses.MISSING:
            kwargs['required'] = True
    else:
        kwargs['type'] = field.type
        if field.default is not dataclasses.MISSING:
            kwargs['default'] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            kwargs['default'] = field.default_factory()
        else:
            kwargs['required'] = True
    parser.add_argument(field_name, *aliases, **kwargs)
    if field.default is True and (field.type is bool or field.type == Optional[bool]):
        bool_kwargs['default'] = False
        parser.add_argument(f'--no_{field.name}', action='store_false', dest=field.name, **bool_kwargs)