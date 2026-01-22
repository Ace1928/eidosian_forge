import collections
import dataclasses
import operator
import re
from typing import Optional
from google.protobuf import struct_pb2
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import metrics
def _value_to_python(value):
    """Converts a google.protobuf.Value to a native Python object."""
    assert isinstance(value, struct_pb2.Value)
    field = value.WhichOneof('kind')
    if field == 'number_value':
        return value.number_value
    elif field == 'string_value':
        return value.string_value
    elif field == 'bool_value':
        return value.bool_value
    else:
        raise ValueError('Unknown struct_pb2.Value oneof field set: %s' % field)