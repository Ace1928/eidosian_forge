import ast
import contextlib
import csv
import inspect
import logging
import os
from argparse import Namespace
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Optional, Type, Union
from warnings import warn
import torch
import yaml
from lightning_utilities.core.apply_func import apply_to_collection
import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning_fabric.utilities.cloud_io import _load as pl_load
from lightning_fabric.utilities.data import AttributeDict
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from pytorch_lightning.accelerators import CUDAAccelerator, MPSAccelerator, XLAAccelerator
from pytorch_lightning.utilities.imports import _OMEGACONF_AVAILABLE
from pytorch_lightning.utilities.migration import pl_legacy_patch
from pytorch_lightning.utilities.migration.utils import _pl_migrate_checkpoint
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.parsing import parse_class_init_keys
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def update_hparams(hparams: dict, updates: dict) -> None:
    """Overrides hparams with new values.

    >>> hparams = {'c': 4}
    >>> update_hparams(hparams, {'a': {'b': 2}, 'c': 1})
    >>> hparams['a']['b'], hparams['c']
    (2, 1)
    >>> update_hparams(hparams, {'a': {'b': 4}, 'c': 7})
    >>> hparams['a']['b'], hparams['c']
    (4, 7)

    Args:
        hparams: the original params and also target object
        updates: new params to be used as update

    """
    for k, v in updates.items():
        if k not in hparams:
            hparams[k] = v
            continue
        if isinstance(v, dict):
            update_hparams(hparams[k], updates[k])
        else:
            hparams.update({k: v})