import logging
import os
from typing import Any, Callable, Dict, Optional, Type, Union, TYPE_CHECKING
import pyarrow.fs
import ray
from ray.air.config import RunConfig
from ray.air._internal.usage import AirEntrypoint
from ray.air.util.node import _force_on_current_node
from ray.train._internal.storage import _exists_at_fs_path, get_fs_and_path
from ray.tune import TuneError
from ray.tune.execution.experiment_state import _ResumeConfig
from ray.tune.experimental.output import (
from ray.tune.result_grid import ResultGrid
from ray.tune.trainable import Trainable
from ray.tune.impl.tuner_internal import TunerInternal, _TUNER_PKL
from ray.tune.tune_config import TuneConfig
from ray.tune.progress_reporter import (
from ray.util import PublicAPI
def _prepare_remote_tuner_for_jupyter_progress_reporting(self):
    run_config: RunConfig = ray.get(self._remote_tuner.get_run_config.remote())
    progress_reporter, string_queue = _prepare_progress_reporter_for_ray_client(run_config.progress_reporter, run_config.verbose)
    run_config.progress_reporter = progress_reporter
    ray.get(self._remote_tuner.set_run_config_and_remote_string_queue.remote(run_config, string_queue))
    return (progress_reporter, string_queue)