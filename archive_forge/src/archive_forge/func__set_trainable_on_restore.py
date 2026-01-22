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
def _set_trainable_on_restore(self, trainable: TrainableType, old_trainable_name: Optional[str]):
    from ray.train.base_trainer import BaseTrainer
    self.trainable = trainable
    assert self.converted_trainable
    self._validate_trainable(trainable=self.converted_trainable, required_trainable_name=old_trainable_name)
    if isinstance(self.trainable, BaseTrainer):
        trainer: BaseTrainer = self.trainable
        if trainer.run_config != RunConfig():
            logger.warning("The Tune experiment will restore using the original run's `RunConfig`. If you made any changes to the `RunConfig` within the Trainer you passed into `Tuner.restore`, they will be ignored in the resumed run.")
        trainer.run_config = self._run_config