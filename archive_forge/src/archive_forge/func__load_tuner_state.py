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
def _load_tuner_state(self, tuner_state: Dict[str, Any]) -> Tuple[Optional[str], Optional[List[str]]]:
    """Loads Tuner state from the previously saved `tuner.pkl`.

        Args:
            tuner_pkl_path: pathlib.Path of the `tuner.pkl` file saved during the
                original Tuner initialization.

        Returns:
            tuple: of `(old_trainable_name, flattened_param_space_keys)` used for
                validating the re-specified `trainable` and `param_space`.
        """
    old_trainable_name = tuner_state.pop('__trainable_name', None)
    flattened_param_space_keys = tuner_state.pop('__flattened_param_space_keys', None)
    self.__setstate__(tuner_state)
    return (old_trainable_name, flattened_param_space_keys)