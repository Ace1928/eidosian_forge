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
def _check_validation_file(self):
    """Checks that the validation file exists at the storage path."""
    valid_file = os.path.join(self.experiment_fs_path, _VALIDATE_STORAGE_MARKER_FILENAME)
    if not _exists_at_fs_path(fs=self.storage_filesystem, fs_path=valid_file):
        raise RuntimeError(f"Unable to set up cluster storage with the following settings:\n{self}\nCheck that all nodes in the cluster have read/write access to the configured storage path. `RunConfig(storage_path)` should be set to a cloud storage URI or a shared filesystem path accessible by all nodes in your cluster ('s3://bucket' or '/mnt/nfs'). A local path on the head node is not accessible by worker nodes. See: https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html")