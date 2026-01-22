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
class _WandbLoggingActor:
    """
    Wandb assumes that each trial's information should be logged from a
    separate process. We use Ray actors as forking multiprocessing
    processes is not supported by Ray and spawn processes run into pickling
    problems.

    We use a queue for the driver to communicate with the logging process.
    The queue accepts the following items:

    - If it's a dict, it is assumed to be a result and will be logged using
      ``wandb.log()``
    - If it's a checkpoint object, it will be saved using ``wandb.log_artifact()``.
    """

    def __init__(self, logdir: str, queue: Queue, exclude: List[str], to_config: List[str], *args, **kwargs):
        import wandb
        self._wandb = wandb
        os.chdir(logdir)
        self.queue = queue
        self._exclude = set(exclude)
        self._to_config = set(to_config)
        self.args = args
        self.kwargs = kwargs
        self._trial_name = self.kwargs.get('name', 'unknown')
        self._logdir = logdir

    def run(self):
        os.environ['WANDB_START_METHOD'] = 'thread'
        run = self._wandb.init(*self.args, **self.kwargs)
        run.config.trial_log_path = self._logdir
        _run_wandb_process_run_info_hook(run)
        while True:
            item_type, item_content = self.queue.get()
            if item_type == _QueueItem.END:
                break
            if item_type == _QueueItem.CHECKPOINT:
                self._handle_checkpoint(item_content)
                continue
            assert item_type == _QueueItem.RESULT
            log, config_update = self._handle_result(item_content)
            try:
                self._wandb.config.update(config_update, allow_val_change=True)
                self._wandb.log(log)
            except urllib.error.HTTPError as e:
                logger.warn('Failed to log result to w&b: {}'.format(str(e)))
        self._wandb.finish()

    def _handle_checkpoint(self, checkpoint_path: str):
        artifact = self._wandb.Artifact(name=f'checkpoint_{self._trial_name}', type='model')
        artifact.add_dir(checkpoint_path)
        self._wandb.log_artifact(artifact)

    def _handle_result(self, result: Dict) -> Tuple[Dict, Dict]:
        config_update = result.get('config', {}).copy()
        log = {}
        flat_result = flatten_dict(result, delimiter='/')
        for k, v in flat_result.items():
            if any((k.startswith(item + '/') or k == item for item in self._exclude)):
                continue
            elif any((k.startswith(item + '/') or k == item for item in self._to_config)):
                config_update[k] = v
            elif not _is_allowed_type(v):
                continue
            else:
                log[k] = v
        config_update.pop('callbacks', None)
        return (log, config_update)