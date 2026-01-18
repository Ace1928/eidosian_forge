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
def validate_module_name(module):
    if not isinstance(module, str):
        raise exceptions.PageError("The first attribute of dash.register_page() must be a string or '__name__'")
    return module