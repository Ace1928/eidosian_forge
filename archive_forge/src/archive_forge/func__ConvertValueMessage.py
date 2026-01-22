import base64
from collections import OrderedDict
import json
import math
from operator import methodcaller
import re
from google.protobuf import descriptor
from google.protobuf import message_factory
from google.protobuf import symbol_database
from google.protobuf.internal import type_checkers
def _ConvertValueMessage(self, value, message, path):
    """Convert a JSON representation into Value message."""
    if isinstance(value, dict):
        self._ConvertStructMessage(value, message.struct_value, path)
    elif isinstance(value, list):
        self._ConvertListValueMessage(value, message.list_value, path)
    elif value is None:
        message.null_value = 0
    elif isinstance(value, bool):
        message.bool_value = value
    elif isinstance(value, str):
        message.string_value = value
    elif isinstance(value, _INT_OR_FLOAT):
        message.number_value = value
    else:
        raise ParseError('Value {0} has unexpected type {1} at {2}'.format(value, type(value), path))