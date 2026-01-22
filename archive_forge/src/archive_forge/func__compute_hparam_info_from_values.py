import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _compute_hparam_info_from_values(self, name, values):
    """Builds an HParamInfo message from the hparam name and list of
        values.

        Args:
          name: string. The hparam name.
          values: list of google.protobuf.Value messages. The list of values for the
            hparam.

        Returns:
          An api_pb2.HParamInfo message.
        """
    result = api_pb2.HParamInfo(name=name, type=api_pb2.DATA_TYPE_UNSET)
    for v in values:
        v_type = _protobuf_value_type(v)
        if not v_type:
            continue
        if result.type == api_pb2.DATA_TYPE_UNSET:
            result.type = v_type
        elif result.type != v_type:
            result.type = api_pb2.DATA_TYPE_STRING
        if result.type == api_pb2.DATA_TYPE_STRING:
            break
    if result.type == api_pb2.DATA_TYPE_UNSET:
        return None
    if result.type == api_pb2.DATA_TYPE_STRING:
        distinct_values = set((_protobuf_value_to_string(v) for v in values if _can_be_converted_to_string(v)))
        result.domain_discrete.extend(distinct_values)
    if result.type == api_pb2.DATA_TYPE_BOOL:
        result.domain_discrete.extend([True, False])
    return result