import copy
import json
import logging
from contextlib import contextmanager
from functools import partial
from numbers import Number
import os
from pathlib import Path
import platform
import re
import time
from typing import Any, Dict, Optional, Sequence, Union, Callable, List, Tuple
import uuid
import ray
from ray.air.constants import (
import ray.cloudpickle as cloudpickle
from ray.exceptions import RayActorError, RayTaskError
from ray.train import Checkpoint, CheckpointConfig
from ray.train.constants import (
from ray.train._internal.checkpoint_manager import _CheckpointManager
from ray.train._internal.session import _FutureTrainingResult, _TrainingResult
from ray.train._internal.storage import StorageContext
from ray.tune import TuneError
from ray.tune.logger import NoopLogger
from ray.tune.registry import get_trainable_cls, validate_trainable
from ray.tune.result import (
from ray.tune.execution.placement_groups import (
from ray.tune.trainable.metadata import _TrainingRunMetadata
from ray.tune.utils.serialization import TuneFunctionDecoder, TuneFunctionEncoder
from ray.tune.utils import date_str, flatten_dict
from ray.util.annotations import DeveloperAPI, Deprecated
from ray._private.utils import binary_to_hex, hex_to_binary
def update_last_result(self, result):
    if self.experiment_tag:
        result.update(experiment_tag=self.experiment_tag)
    self.set_location(_Location(result.get(NODE_IP), result.get(PID)))
    self.run_metadata.last_result = result
    self.run_metadata.last_result_time = time.time()
    metric_result = self.last_result.copy()
    for remove_metric in DEBUG_METRICS:
        metric_result.pop(remove_metric, None)
    for metric, value in flatten_dict(metric_result).items():
        if isinstance(value, Number):
            self.run_metadata.update_metric(metric, value, step=result.get('training_iteration'))