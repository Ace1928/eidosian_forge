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
def _build_data_provider_filter(col_param):
    """Builds HyperparameterFilter from ColParam.

    Args:
      col_param: ColParam that possibly contains filter information.

    Returns:
      None if col_param does not specify filter information.
    """
    if col_param.HasField('filter_regexp'):
        filter_type = provider.HyperparameterFilterType.REGEX
        fltr = col_param.filter_regexp
    elif col_param.HasField('filter_interval'):
        filter_type = provider.HyperparameterFilterType.INTERVAL
        fltr = (col_param.filter_interval.min_value, col_param.filter_interval.max_value)
    elif col_param.HasField('filter_discrete'):
        filter_type = provider.HyperparameterFilterType.DISCRETE
        fltr = [_value_to_python(b) for b in col_param.filter_discrete.values]
    else:
        return None
    return provider.HyperparameterFilter(hyperparameter_name=col_param.hparam, filter_type=filter_type, filter=fltr)