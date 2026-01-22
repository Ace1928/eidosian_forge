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
class _MetricStats:
    """A simple class to hold metric stats used in calculating metric averages.

    Used in _set_avg_session_metrics(). See the comments in that function
    for more details.

    Attributes:
      total: int. The sum of the metric measurements seen so far.
      count: int. The number of largest-step measuremens seen so far.
      total_step: int. The sum of the steps at which the measurements were taken
      total_wall_time_secs: float. The sum of the wall_time_secs at
          which the measurements were taken.
    """
    __slots__ = ['total', 'count', 'total_step', 'total_wall_time_secs']

    def __init__(self):
        self.total = 0
        self.count = 0
        self.total_step = 0
        self.total_wall_time_secs = 0.0