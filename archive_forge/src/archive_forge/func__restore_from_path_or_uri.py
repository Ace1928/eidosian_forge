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
def _restore_from_path_or_uri(self, path_or_uri: str, trainable: TrainableTypeOrTrainer, overwrite_param_space: Optional[Dict[str, Any]], resume_config: _ResumeConfig, storage_filesystem: Optional[pyarrow.fs.FileSystem]):
    fs, fs_path = get_fs_and_path(path_or_uri, storage_filesystem)
    with fs.open_input_file(os.path.join(fs_path, _TUNER_PKL)) as f:
        tuner_state = pickle.loads(f.readall())
    old_trainable_name, flattened_param_space_keys = self._load_tuner_state(tuner_state)
    self._set_trainable_on_restore(trainable=trainable, old_trainable_name=old_trainable_name)
    self._set_param_space_on_restore(param_space=overwrite_param_space, flattened_param_space_keys=flattened_param_space_keys)
    path_or_uri_obj = URI(path_or_uri)
    self._run_config.name = path_or_uri_obj.name
    self._run_config.storage_path = str(path_or_uri_obj.parent)
    self._local_experiment_dir, self._experiment_dir_name = self.setup_create_experiment_checkpoint_dir(self.converted_trainable, self._run_config)
    try:
        self._experiment_analysis = ExperimentAnalysis(experiment_checkpoint_path=path_or_uri, default_metric=self._tune_config.metric, default_mode=self._tune_config.mode)
    except Exception:
        self._experiment_analysis = None
    self._resume_config = resume_config
    self._is_restored = True