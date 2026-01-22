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
def _get_full_model_names_from_exp_structure(cls, exp_structure: Dict[str, Any], namespace: str) -> Set[str]:
    """Returns all paths to properties which were already logged in `namespace`"""
    structure_keys: List[str] = namespace.split(cls.LOGGER_JOIN_CHAR)
    for key in structure_keys:
        exp_structure = exp_structure[key]
    uploaded_models_dict = exp_structure
    return set(cls._dict_paths(uploaded_models_dict))