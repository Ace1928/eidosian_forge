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
@staticmethod
def __max_ckpt_version_in_folder(dir_path: _PATH, name_key: str='ckpt_') -> Optional[int]:
    """List up files in `dir_path` with `name_key`, then yield maximum suffix number.

        Args:
            dir_path: path of directory which may contain files whose name include `name_key`
            name_key: file name prefix
        Returns:
            None if no-corresponding-file else maximum suffix number

        """
    fs, uri = url_to_fs(str(dir_path))
    if not fs.exists(dir_path):
        return None
    files = [os.path.basename(f['name']) for f in fs.listdir(uri)]
    files = [x for x in files if name_key in x]
    if len(files) == 0:
        return None
    ckpt_vs = []
    for name in files:
        name = name.split(name_key)[-1]
        name = re.sub('[^0-9]', '', name)
        ckpt_vs.append(int(name))
    return max(ckpt_vs)