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
def _cleanup_logging_actors(self, timeout: int=0, kill_on_timeout: bool=False):
    """Clean up logging actors that have finished uploading to wandb.
        Waits for `timeout` seconds to collect finished logging actors.

        Args:
            timeout: The number of seconds to wait. Defaults to 0 to clean up
                any immediate logging actors during the run.
                This is set to a timeout threshold to wait for pending uploads
                on experiment end.
            kill_on_timeout: Whether or not to kill and cleanup the logging actor if
                it hasn't finished within the timeout.
        """
    futures = list(self._trial_logging_futures.values())
    done, remaining = ray.wait(futures, num_returns=len(futures), timeout=timeout)
    for ready_future in done:
        finished_trial = self._logging_future_to_trial.pop(ready_future)
        self._cleanup_logging_actor(finished_trial)
    if kill_on_timeout:
        for remaining_future in remaining:
            trial = self._logging_future_to_trial.pop(remaining_future)
            self._cleanup_logging_actor(trial)