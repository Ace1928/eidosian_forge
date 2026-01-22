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
def _get_preprocessor_error(name: str):
    return AttributeError(f'The new `ray.train.Checkpoint` class does not support `{name}()`. To include preprocessor information in checkpoints, pass it as metadata in the <Framework>Trainer constructor.\nExample: `TorchTrainer(..., metadata={{...}})`.\nAfter training, access it in the checkpoint via `checkpoint.get_metadata()`. See here: https://docs.ray.io/en/master/train/user-guides/data-loading-preprocessing.html#preprocessing-structured-data')