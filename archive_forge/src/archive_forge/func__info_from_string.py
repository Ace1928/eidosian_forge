import base64
import dataclasses
import datetime
import errno
import json
import os
import subprocess
import tempfile
import time
import typing
from typing import Optional
from tensorboard import version
from tensorboard.util import tb_logging
def _info_from_string(info_string):
    """Parse a `TensorBoardInfo` object from its string representation.

    Args:
      info_string: A string representation of a `TensorBoardInfo`, as
        produced by a previous call to `_info_to_string`.

    Returns:
      A `TensorBoardInfo` value.

    Raises:
      ValueError: If the provided string is not valid JSON, or if it is
        missing any required fields, or if any field is of incorrect type.
    """
    field_name_to_type = typing.get_type_hints(TensorBoardInfo)
    try:
        json_value = json.loads(info_string)
    except ValueError:
        raise ValueError('invalid JSON: %r' % (info_string,))
    if not isinstance(json_value, dict):
        raise ValueError('not a JSON object: %r' % (json_value,))
    expected_keys = frozenset(field_name_to_type.keys())
    actual_keys = frozenset(json_value)
    missing_keys = expected_keys - actual_keys
    if missing_keys:
        raise ValueError('TensorBoardInfo missing keys: %r' % (sorted(missing_keys),))
    fields = {}
    for key, field_type in field_name_to_type.items():
        if not isinstance(json_value[key], field_type):
            raise ValueError('expected %r of type %s, but found: %r' % (key, field_type, json_value[key]))
        fields[key] = json_value[key]
    return TensorBoardInfo(**fields)