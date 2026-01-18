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
def validate_multi_return(output_lists, output_values, callback_id):
    if not isinstance(output_values, (list, tuple)):
        raise exceptions.InvalidCallbackReturnValue(dedent(f'\n                The callback {callback_id} is a multi-output.\n                Expected the output type to be a list or tuple but got:\n                {output_values!r}.\n                '))
    if len(output_values) != len(output_lists):
        raise exceptions.InvalidCallbackReturnValue(f'\n            Invalid number of output values for {callback_id}.\n            Expected {len(output_lists)}, got {len(output_values)}\n            ')
    for i, output_spec in enumerate(output_lists):
        if isinstance(output_spec, list):
            output_value = output_values[i]
            if not isinstance(output_value, (list, tuple)):
                raise exceptions.InvalidCallbackReturnValue(dedent(f'\n                        The callback {callback_id} output {i} is a wildcard multi-output.\n                        Expected the output type to be a list or tuple but got:\n                        {output_value!r}.\n                        output spec: {output_spec!r}\n                        '))
            if len(output_value) != len(output_spec):
                raise exceptions.InvalidCallbackReturnValue(dedent(f'\n                        Invalid number of output values for {callback_id} item {i}.\n                        Expected {len(output_spec)}, got {len(output_value)}\n                        output spec: {output_spec!r}\n                        output value: {output_value!r}\n                        '))