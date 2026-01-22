import contextlib
import logging
import os
from argparse import Namespace
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set, Union
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.logger import _add_prefix, _convert_params, _sanitize_callable_params
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.utilities.rank_zero import rank_zero_only
@staticmethod
def _get_full_model_name(model_path: str, checkpoint_callback: Checkpoint) -> str:
    """Returns model name which is string `model_path` appended to `checkpoint_callback.dirpath`."""
    if hasattr(checkpoint_callback, 'dirpath'):
        model_path = os.path.normpath(model_path)
        expected_model_path = os.path.normpath(checkpoint_callback.dirpath)
        if not model_path.startswith(expected_model_path):
            raise ValueError(f'{model_path} was expected to start with {expected_model_path}.')
        filepath, _ = os.path.splitext(model_path[len(expected_model_path) + 1:])
        return filepath.replace(os.sep, '/')
    return model_path.replace(os.sep, '/')