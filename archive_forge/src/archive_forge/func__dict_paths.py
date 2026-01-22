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
@classmethod
def _dict_paths(cls, d: Dict[str, Any], path_in_build: Optional[str]=None) -> Generator:
    for k, v in d.items():
        path = f'{path_in_build}/{k}' if path_in_build is not None else k
        if not isinstance(v, dict):
            yield path
        else:
            yield from cls._dict_paths(v, path)