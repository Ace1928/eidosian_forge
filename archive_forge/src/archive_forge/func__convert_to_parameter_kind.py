import inspect
import logging
from inspect import Parameter
from ray._private.inspect_util import is_cython
def _convert_to_parameter_kind(value):
    if value == 0:
        return Parameter.POSITIONAL_ONLY
    if value == 1:
        return Parameter.POSITIONAL_OR_KEYWORD
    if value == 2:
        return Parameter.VAR_POSITIONAL
    if value == 3:
        return Parameter.KEYWORD_ONLY
    if value == 4:
        return Parameter.VAR_KEYWORD