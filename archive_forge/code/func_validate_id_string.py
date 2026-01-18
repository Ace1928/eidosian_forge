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
def validate_id_string(arg):
    arg_id = arg.component_id
    invalid_chars = '.{'
    invalid_found = [x for x in invalid_chars if x in arg_id]
    if invalid_found:
        raise exceptions.InvalidComponentIdError(f'\n            The element `{arg_id}` contains `{'`, `'.join(invalid_found)}` in its ID.\n            Characters `{'`, `'.join(invalid_chars)}` are not allowed in IDs.\n            ')