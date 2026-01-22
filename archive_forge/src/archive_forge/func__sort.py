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
def _sort(self, session_groups, extractors):
    """Sorts 'session_groups' in place according to _request.col_params."""
    session_groups.sort(key=operator.attrgetter('name'))
    for col_param, extractor in reversed(list(zip(self._request.col_params, extractors))):
        if col_param.order == api_pb2.ORDER_UNSPECIFIED:
            continue
        if col_param.order == api_pb2.ORDER_ASC:
            session_groups.sort(key=_create_key_func(extractor, none_is_largest=not col_param.missing_values_first))
        elif col_param.order == api_pb2.ORDER_DESC:
            session_groups.sort(key=_create_key_func(extractor, none_is_largest=col_param.missing_values_first), reverse=True)
        else:
            raise error.HParamsError('Unknown col_param.order given: %s' % col_param)