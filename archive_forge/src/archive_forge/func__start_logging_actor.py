import enum
import os
import pickle
import urllib
import warnings
import numpy as np
from numbers import Number
import pyarrow.fs
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import ray
from ray import logger
from ray.air import session
from ray.air._internal import usage as air_usage
from ray.air.util.node import _force_on_current_node
from ray.tune.logger import LoggerCallback
from ray.tune.utils import flatten_dict
from ray.tune.experiment import Trial
from ray.train._internal.syncer import DEFAULT_SYNC_TIMEOUT
from ray._private.storage import _load_class
from ray.util import PublicAPI
from ray.util.queue import Queue
def _start_logging_actor(self, trial: 'Trial', exclude_results: List[str], **wandb_init_kwargs):
    if not self._remote_logger_class:
        env_vars = {}
        if WANDB_ENV_VAR in os.environ:
            env_vars[WANDB_ENV_VAR] = os.environ[WANDB_ENV_VAR]
        self._remote_logger_class = ray.remote(num_cpus=0, **_force_on_current_node(), runtime_env={'env_vars': env_vars}, max_restarts=-1, max_task_retries=-1)(self._logger_actor_cls)
    self._trial_queues[trial] = Queue(actor_options={'num_cpus': 0, **_force_on_current_node(), 'max_restarts': -1, 'max_task_retries': -1})
    self._trial_logging_actors[trial] = self._remote_logger_class.remote(logdir=trial.local_path, queue=self._trial_queues[trial], exclude=exclude_results, to_config=self.AUTO_CONFIG_KEYS, **wandb_init_kwargs)
    logging_future = self._trial_logging_actors[trial].run.remote()
    self._trial_logging_futures[trial] = logging_future
    self._logging_future_to_trial[logging_future] = trial