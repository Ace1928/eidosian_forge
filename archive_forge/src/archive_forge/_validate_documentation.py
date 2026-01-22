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

    This validation is for security and internal debugging, not for users,
    so the messages are not intended to be clear.
    `output` comes from the callback definition, `output_spec` from the request.
    