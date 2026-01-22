import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _protobuf_value_type(value):
    """Returns the type of the google.protobuf.Value message as an
    api.DataType.

    Returns None if the type of 'value' is not one of the types supported in
    api_pb2.DataType.

    Args:
      value: google.protobuf.Value message.
    """
    if value.HasField('number_value'):
        return api_pb2.DATA_TYPE_FLOAT64
    if value.HasField('string_value'):
        return api_pb2.DATA_TYPE_STRING
    if value.HasField('bool_value'):
        return api_pb2.DATA_TYPE_BOOL
    return None