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
def _choose_run_config(self, tuner_run_config: Optional[RunConfig], trainer: 'BaseTrainer', param_space: Optional[Dict[str, Any]]) -> RunConfig:
    """Chooses which `RunConfig` to use when multiple can be passed in
        through a Trainer or the Tuner itself.

        Args:
            tuner_run_config: The run config passed into the Tuner constructor.
            trainer: The Trainer instance to use with Tune, which may have
                a RunConfig specified by the user.
            param_space: The param space passed to the Tuner.

        Raises:
            ValueError: if the `run_config` is specified as a hyperparameter.
        """
    if param_space and 'run_config' in param_space:
        raise ValueError('`RunConfig` cannot be tuned as part of the `param_space`! Move the run config to be a parameter of the `Tuner`: Tuner(..., run_config=RunConfig(...))')
    if tuner_run_config and trainer.run_config != RunConfig():
        logger.info(f'A `RunConfig` was passed to both the `Tuner` and the `{trainer.__class__.__name__}`. The run config passed to the `Tuner` is the one that will be used.')
        return tuner_run_config
    if not tuner_run_config:
        return trainer.run_config
    return tuner_run_config