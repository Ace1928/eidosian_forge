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
def _open_logfiles(self, stdout_file, stderr_file):
    """Create loggers. Open stdout and stderr logfiles."""
    if stdout_file:
        stdout_path = (Path(self._logdir) / stdout_file).expanduser().as_posix()
        self._stdout_fp = open(stdout_path, 'a+')
        self._stdout_stream = Tee(sys.stdout, self._stdout_fp)
        self._stdout_context = redirect_stdout(self._stdout_stream)
        self._stdout_context.__enter__()
    if stderr_file:
        stderr_path = (Path(self._logdir) / stderr_file).expanduser().as_posix()
        self._stderr_fp = open(stderr_path, 'a+')
        self._stderr_stream = Tee(sys.stderr, self._stderr_fp)
        self._stderr_context = redirect_stderr(self._stderr_stream)
        self._stderr_context.__enter__()
        formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(filename)s: %(lineno)d  %(message)s')
        self._stderr_logging_handler = logging.StreamHandler(self._stderr_fp)
        self._stderr_logging_handler.setFormatter(formatter)
        ray.logger.addHandler(self._stderr_logging_handler)