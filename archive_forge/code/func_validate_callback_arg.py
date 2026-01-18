import sys
from collections.abc import MutableSequence
import re
from textwrap import dedent
from keyword import iskeyword
import flask
from ._grouping import grouping_len, map_grouping
from .development.base_component import Component
from . import exceptions
from ._utils import (
def validate_callback_arg(arg):
    if not isinstance(getattr(arg, 'component_property', None), str):
        raise exceptions.IncorrectTypeException(dedent(f'\n                component_property must be a string, found {arg.component_property!r}\n                '))
    if hasattr(arg, 'component_event'):
        raise exceptions.NonExistentEventException('\n            Events have been removed.\n            Use the associated property instead.\n            ')
    if isinstance(arg.component_id, dict):
        validate_id_dict(arg)
    elif isinstance(arg.component_id, str):
        validate_id_string(arg)
    else:
        raise exceptions.IncorrectTypeException(dedent(f'\n                component_id must be a string or dict, found {arg.component_id!r}\n                '))