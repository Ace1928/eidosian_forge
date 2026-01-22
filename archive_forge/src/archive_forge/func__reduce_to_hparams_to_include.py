import collections
import dataclasses
import operator
import re
from typing import Optional
from google.protobuf import struct_pb2
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import backend_context as backend_context_lib
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import metrics
from tensorboard.plugins.hparams import plugin_data_pb2
def _reduce_to_hparams_to_include(session_groups, col_params):
    """Removes hparams from session_groups that should not be included.

    Args:
      session_groups: A collection of `SessionGroup` protos, which will be
        modified in place.
      col_params: A collection of `ColParams` protos.
    """
    hparams_to_include = _get_hparams_to_include(col_params)
    for session_group in session_groups:
        new_hparams = {hparam: value for hparam, value in session_group.hparams.items() if hparam in hparams_to_include}
        session_group.ClearField('hparams')
        for hparam, value in new_hparams.items():
            session_group.hparams[hparam].CopyFrom(value)