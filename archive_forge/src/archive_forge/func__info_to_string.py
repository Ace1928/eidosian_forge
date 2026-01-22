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
def _info_to_string(info):
    """Convert a `TensorBoardInfo` to string form to be stored on disk.

    The format returned by this function is opaque and should only be
    interpreted by `_info_from_string`.

    Args:
      info: A valid `TensorBoardInfo` object.

    Raises:
      ValueError: If any field on `info` is not of the correct type.

    Returns:
      A string representation of the provided `TensorBoardInfo`.
    """
    field_name_to_type = typing.get_type_hints(TensorBoardInfo)
    for key, field_type in field_name_to_type.items():
        if not isinstance(getattr(info, key), field_type):
            raise ValueError('expected %r of type %s, but found: %r' % (key, field_type, getattr(info, key)))
    if info.version != version.VERSION:
        raise ValueError("expected 'version' to be %r, but found: %r" % (version.VERSION, info.version))
    json_value = dataclasses.asdict(info)
    return json.dumps(json_value, sort_keys=True, indent=4)