import logging
import os
import re
import shutil
import time
import warnings
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Set
from weakref import proxy
import torch
import yaml
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import _is_dir, _is_local_file_protocol, get_filesystem
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT
@staticmethod
def _link_checkpoint(trainer: 'pl.Trainer', filepath: str, linkpath: str) -> None:
    if trainer.is_global_zero:
        if os.path.islink(linkpath) or os.path.isfile(linkpath):
            os.remove(linkpath)
        elif os.path.isdir(linkpath):
            shutil.rmtree(linkpath)
        try:
            os.symlink(os.path.relpath(filepath, os.path.dirname(linkpath)), linkpath)
        except OSError:
            shutil.copy(filepath, linkpath)
    trainer.strategy.barrier()