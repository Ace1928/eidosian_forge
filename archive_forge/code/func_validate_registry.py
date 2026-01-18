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
def validate_registry(registry):
    for page in registry.values():
        if 'layout' not in page:
            raise exceptions.NoLayoutException(f'No layout in module `{page['module']}` in dash.page_registry')
        if page['module'] == '__main__':
            raise Exception('\n                When registering pages from app.py, `__name__` is not a valid module name.  Use a string instead.\n                For example, `dash.register_page("my_module_name")`, rather than `dash.register_page(__name__)`\n                ')