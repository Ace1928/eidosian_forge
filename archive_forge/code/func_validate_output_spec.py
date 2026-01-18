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
def validate_output_spec(output, output_spec, Output):
    """
    This validation is for security and internal debugging, not for users,
    so the messages are not intended to be clear.
    `output` comes from the callback definition, `output_spec` from the request.
    """
    if not isinstance(output, (list, tuple)):
        output, output_spec = ([output], [output_spec])
    elif len(output) != len(output_spec):
        raise exceptions.CallbackException('Wrong length output_spec')
    for outi, speci in zip(output, output_spec):
        speci_list = speci if isinstance(speci, (list, tuple)) else [speci]
        for specij in speci_list:
            if not Output(specij['id'], clean_property_name(specij['property'])) == outi:
                raise exceptions.CallbackException('Output does not match callback definition')