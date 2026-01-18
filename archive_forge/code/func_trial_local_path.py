import dataclasses
import fnmatch
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type, Union
from ray._private.storage import _get_storage_uri
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.syncer import SyncConfig, Syncer, _BackgroundSyncer
from ray.train.constants import _get_defaults_results_dir
@property
def trial_local_path(self) -> str:
    """The local filesystem path to the trial directory.

        Raises a ValueError if `trial_dir_name` is not set beforehand.
        """
    if self.trial_dir_name is None:
        raise RuntimeError('Should not access `trial_local_path` without setting `trial_dir_name`')
    return Path(self.experiment_local_path, self.trial_dir_name).as_posix()