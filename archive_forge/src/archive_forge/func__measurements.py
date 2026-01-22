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
def _measurements(session_group, metric_name):
    """A generator for the values of the metric across the sessions in the
    group.

    Args:
      session_group: A SessionGroup protobuffer.
      metric_name: A MetricName protobuffer.
    Yields:
      The next metric value wrapped in a _Measurement instance.
    """
    for session_index, session in enumerate(session_group.sessions):
        metric_value = _find_metric_value(session, metric_name)
        if not metric_value:
            continue
        yield _Measurement(metric_value, session_index)