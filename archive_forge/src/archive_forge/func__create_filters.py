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
def _create_filters(col_params, extractors):
    """Creates filters for the given col_params.

    Args:
      col_params: List of ListSessionGroupsRequest.ColParam protobufs.
      extractors: list of extractor functions of the same length as col_params.
        Each element should extract the column described by the corresponding
        element of col_params.
    Returns:
      A list of filter functions. Each corresponding to a single
      col_params.filter oneof field of _request
    """
    result = []
    for col_param, extractor in zip(col_params, extractors):
        a_filter = _create_filter(col_param, extractor)
        if a_filter:
            result.append(a_filter)
    return result