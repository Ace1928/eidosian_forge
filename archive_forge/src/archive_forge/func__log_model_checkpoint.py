import functools
import importlib.metadata
import importlib.util
import json
import numbers
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union
import numpy as np
from .. import __version__ as version
from ..utils import flatten_dict, is_datasets_available, is_pandas_available, is_torch_available, logging
from ..trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from ..trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from ..training_args import ParallelMode  # noqa: E402
from ..utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402
def _log_model_checkpoint(self, source_directory: str, checkpoint: str):
    target_path = relative_path = os.path.join(source_directory, checkpoint)
    if self._volatile_checkpoints_dir is not None:
        consistent_checkpoint_path = os.path.join(self._volatile_checkpoints_dir, checkpoint)
        try:
            cpkt_path = relative_path.replace('..', '').lstrip(os.path.sep)
            copy_path = os.path.join(consistent_checkpoint_path, cpkt_path)
            shutil.copytree(relative_path, copy_path)
            target_path = consistent_checkpoint_path
        except IOError as e:
            logger.warning("NeptuneCallback was unable to made a copy of checkpoint due to I/O exception: '{}'. Could fail trying to upload.".format(e))
    self._metadata_namespace[self._target_checkpoints_namespace].upload_files(target_path)
    if self._should_clean_recently_uploaded_checkpoint and self._recent_checkpoint_path is not None:
        self._metadata_namespace[self._target_checkpoints_namespace].delete_files(self._recent_checkpoint_path)
    self._recent_checkpoint_path = relative_path