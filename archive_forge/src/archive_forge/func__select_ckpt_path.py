import logging
import os
import re
from typing import Any, Dict, Optional
import torch
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from torch import Tensor
import pytorch_lightning as pl
from lightning_fabric.plugins.environments.slurm import SLURMEnvironment
from lightning_fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.precision import MixedPrecision
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _OMEGACONF_AVAILABLE
from pytorch_lightning.utilities.migration import pl_legacy_patch
from pytorch_lightning.utilities.migration.utils import _pl_migrate_checkpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _select_ckpt_path(self, state_fn: TrainerFn, ckpt_path: Optional[_PATH], model_provided: bool, model_connected: bool) -> Optional[_PATH]:
    """Called by the ``Trainer`` to select the checkpoint path source."""
    if self._user_managed:
        if ckpt_path:
            rank_zero_warn(f'`trainer.ckpt_path = {self._ckpt_path!r}` was called but then you passed `trainer.fit(ckpt_path={ckpt_path!r})`. The latter will be loaded.')
            self._ckpt_path = None
            self._user_managed = False
            ckpt_path = self._parse_ckpt_path(state_fn, ckpt_path, model_provided=model_provided, model_connected=model_connected)
        else:
            ckpt_path = self._ckpt_path
    else:
        ckpt_path = self._parse_ckpt_path(state_fn, ckpt_path, model_provided=model_provided, model_connected=model_connected)
    return ckpt_path