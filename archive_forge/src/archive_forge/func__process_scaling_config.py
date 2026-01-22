import copy
import io
import os
import math
import logging
from pathlib import Path
from typing import (
import pyarrow.fs
import ray.cloudpickle as pickle
from ray.util import inspect_serializability
from ray.air._internal.uri_utils import URI
from ray.air._internal.usage import AirEntrypoint
from ray.air.config import RunConfig, ScalingConfig
from ray.train._internal.storage import StorageContext, get_fs_and_path
from ray.tune import Experiment, TuneError, ExperimentAnalysis
from ray.tune.execution.experiment_state import _ResumeConfig
from ray.tune.tune import _Config
from ray.tune.registry import is_function_trainable
from ray.tune.result import _get_defaults_results_dir
from ray.tune.result_grid import ResultGrid
from ray.tune.trainable import Trainable
from ray.tune.tune import run
from ray.tune.tune_config import TuneConfig
from ray.tune.utils import flatten_dict
def _process_scaling_config(self) -> None:
    """Converts ``self._param_space["scaling_config"]`` to a dict.

        The dict is converted back to a dataclass by the Trainer, after the
        Tune search specification is resolved.
        """
    scaling_config = self._param_space.get('scaling_config')
    if not isinstance(scaling_config, ScalingConfig):
        return
    self._param_space['scaling_config'] = scaling_config.__dict__.copy()