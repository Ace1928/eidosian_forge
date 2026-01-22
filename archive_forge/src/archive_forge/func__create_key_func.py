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
def _create_key_func(extractor, none_is_largest):
    """Returns a key_func to be used in list.sort().

    Returns a key_func to be used in list.sort() that sorts session groups
    by the value extracted by extractor. 'None' extracted values will either
    be considered largest or smallest as specified by the "none_is_largest"
    boolean parameter.

    Args:
      extractor: An extractor function that extract the key from the session
        group.
      none_is_largest: bool. If true treats 'None's as largest; otherwise
        smallest.
    """
    if none_is_largest:

        def key_func_none_is_largest(session_group):
            value = extractor(session_group)
            return (value is None, value)
        return key_func_none_is_largest

    def key_func_none_is_smallest(session_group):
        value = extractor(session_group)
        return (value is not None, value)
    return key_func_none_is_smallest