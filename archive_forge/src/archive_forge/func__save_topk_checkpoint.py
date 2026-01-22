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
def _save_topk_checkpoint(self, trainer: 'pl.Trainer', monitor_candidates: Dict[str, Tensor]) -> None:
    if self.save_top_k == 0:
        return
    if self.monitor is not None:
        if self.monitor not in monitor_candidates:
            m = f'`ModelCheckpoint(monitor={self.monitor!r})` could not find the monitored key in the returned metrics: {list(monitor_candidates)}. HINT: Did you call `log({self.monitor!r}, value)` in the `LightningModule`?'
            if trainer.fit_loop.epoch_loop.val_loop._has_run:
                raise MisconfigurationException(m)
            warning_cache.warn(m)
        self._save_monitor_checkpoint(trainer, monitor_candidates)
    else:
        self._save_none_monitor_checkpoint(trainer, monitor_candidates)