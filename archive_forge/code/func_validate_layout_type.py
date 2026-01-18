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
def validate_layout_type(value):
    if not isinstance(value, (Component, patch_collections_abc('Callable'))):
        raise exceptions.NoLayoutException('\n            Layout must be a single dash component\n            or a function that returns a dash component.\n            Cannot be a tuple (are there any trailing commas?)\n            ')