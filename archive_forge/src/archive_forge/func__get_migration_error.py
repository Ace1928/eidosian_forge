import contextlib
import glob
import json
import logging
import os
import platform
import shutil
import tempfile
import traceback
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union
import pyarrow.fs
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.storage import _download_from_fs_path, _exists_at_fs_path
from ray.util.annotations import PublicAPI
def _get_migration_error(name: str):
    return AttributeError(f"The new `ray.train.Checkpoint` class does not support `{name}()`. Instead, only directories are supported.\n\nExample to store a dictionary in a checkpoint:\n\nimport os, tempfile\nimport ray.cloudpickle as pickle\nfrom ray import train\nfrom ray.train import Checkpoint\n\nwith tempfile.TemporaryDirectory() as checkpoint_dir:\n  with open(os.path.join(checkpoint_dir, 'data.pkl'), 'wb') as fp:\n    pickle.dump({{'data': 'value'}}, fp)\n\n  checkpoint = Checkpoint.from_directory(checkpoint_dir)\n  train.report(..., checkpoint=checkpoint)\n\nExample to load a dictionary from a checkpoint:\n\nif train.get_checkpoint():\n  with train.get_checkpoint().as_directory() as checkpoint_dir:\n    with open(os.path.join(checkpoint_dir, 'data.pkl'), 'rb') as fp:\n      data = pickle.load(fp)")