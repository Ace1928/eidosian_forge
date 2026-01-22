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
@property
def _hpc_resume_path(self) -> Optional[str]:
    dir_path_hpc = self.trainer.default_root_dir
    dir_path_hpc = str(dir_path_hpc)
    fs, path = url_to_fs(dir_path_hpc)
    if not _is_dir(fs, path):
        return None
    max_version = self.__max_ckpt_version_in_folder(dir_path_hpc, 'hpc_ckpt_')
    if max_version is not None:
        if isinstance(fs, LocalFileSystem):
            return os.path.join(dir_path_hpc, f'hpc_ckpt_{max_version}.ckpt')
        return dir_path_hpc + fs.sep + f'hpc_ckpt_{max_version}.ckpt'
    return None