import copy
from datetime import datetime
import logging
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import ray
import ray.cloudpickle as ray_pickle
from ray.air._internal.util import skip_exceptions, exception_cause
from ray.air.constants import (
from ray.train._internal.checkpoint_manager import _TrainingResult
from ray.train._internal.storage import StorageContext, _exists_at_fs_path
from ray.train import Checkpoint
from ray.tune.result import (
from ray.tune.utils import UtilMonitor
from ray.tune.utils.log import disable_ipython
from ray.tune.utils.util import Tee
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.util.annotations import DeveloperAPI, PublicAPI
def _create_logger(self, config: Dict[str, Any], logger_creator: Callable[[Dict[str, Any]], 'Logger']=None):
    """Create logger from logger creator.

        Sets _logdir and _result_logger.

        `_logdir` is the **per trial** directory for the Trainable.
        """
    if logger_creator:
        self._result_logger = logger_creator(config)
        self._logdir = self._result_logger.logdir
    else:
        from ray.tune.logger import UnifiedLogger
        logdir_prefix = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        ray._private.utils.try_to_create_directory(DEFAULT_RESULTS_DIR)
        self._logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=DEFAULT_RESULTS_DIR)
        self._result_logger = UnifiedLogger(config, self._logdir, loggers=None)