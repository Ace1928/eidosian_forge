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
@dataclasses.dataclass(frozen=True)
class _MetricIdentifier:
    """An identifier for a metric.

    As protobuffers are mutable we can't use MetricName directly as a dict's key.
    Instead, we represent MetricName protocol buffer as an immutable dataclass.

    Attributes:
      group: Metric group corresponding to the dataset on which the model was
        evaluated.
      tag: String tag associated with the metric.
    """
    group: str
    tag: str