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
def train_buffered(self, buffer_time_s: float, max_buffer_length: int=1000):
    """Runs multiple iterations of training.

        Calls ``train()`` internally. Collects and combines multiple results.
        This function will run ``self.train()`` repeatedly until one of
        the following conditions is met: 1) the maximum buffer length is
        reached, 2) the maximum buffer time is reached, or 3) a checkpoint
        was created. Even if the maximum time is reached, it will always
        block until at least one result is received.

        Args:
            buffer_time_s: Maximum time to buffer. The next result
                received after this amount of time has passed will return
                the whole buffer.
            max_buffer_length: Maximum number of results to buffer.

        """
    results = []
    now = time.time()
    send_buffer_at = now + buffer_time_s
    while now < send_buffer_at or not results:
        result = self.train()
        results.append(result)
        if result.get(DONE, False):
            break
        elif result.get(SHOULD_CHECKPOINT, False):
            break
        elif result.get(RESULT_DUPLICATE):
            break
        elif len(results) >= max_buffer_length:
            break
        now = time.time()
    return results