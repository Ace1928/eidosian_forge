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
def validate_id_dict(arg):
    arg_id = arg.component_id
    for k in arg_id:
        if not isinstance(k, str):
            raise exceptions.IncorrectTypeException(dedent(f'\n                    Wildcard ID keys must be non-empty strings,\n                    found {k!r} in id {arg_id!r}\n                    '))