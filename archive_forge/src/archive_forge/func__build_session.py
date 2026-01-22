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
def _build_session(self, experiment, name, start_info, end_info, all_metric_evals):
    """Builds a session object."""
    assert start_info is not None
    result = api_pb2.Session(name=name, start_time_secs=start_info.start_time_secs, model_uri=start_info.model_uri, metric_values=self._build_session_metric_values(experiment, name, all_metric_evals), monitor_url=start_info.monitor_url)
    if end_info is not None:
        result.status = end_info.status
        result.end_time_secs = end_info.end_time_secs
    return result