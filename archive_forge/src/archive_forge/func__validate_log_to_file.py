import copy
import datetime
from functools import partial
import logging
from pathlib import Path
from pickle import PicklingError
import pprint as pp
import traceback
from typing import (
import ray
from ray.exceptions import RpcError
from ray.train import CheckpointConfig, SyncConfig
from ray.train._internal.storage import StorageContext
from ray.tune.error import TuneError
from ray.tune.registry import register_trainable, is_function_trainable
from ray.tune.stopper import CombinedStopper, FunctionStopper, Stopper, TimeoutStopper
from ray.util.annotations import DeveloperAPI, Deprecated
def _validate_log_to_file(log_to_file):
    """Validate ``train.RunConfig``'s ``log_to_file`` parameter. Return
    validated relative stdout and stderr filenames."""
    if not log_to_file:
        stdout_file = stderr_file = None
    elif isinstance(log_to_file, bool) and log_to_file:
        stdout_file = 'stdout'
        stderr_file = 'stderr'
    elif isinstance(log_to_file, str):
        stdout_file = stderr_file = log_to_file
    elif isinstance(log_to_file, Sequence):
        if len(log_to_file) != 2:
            raise ValueError('If you pass a Sequence to `log_to_file` it has to have a length of 2 (for stdout and stderr, respectively). The Sequence you passed has length {}.'.format(len(log_to_file)))
        stdout_file, stderr_file = log_to_file
    else:
        raise ValueError('You can pass a boolean, a string, or a Sequence of length 2 to `log_to_file`, but you passed something else ({}).'.format(type(log_to_file)))
    return (stdout_file, stderr_file)