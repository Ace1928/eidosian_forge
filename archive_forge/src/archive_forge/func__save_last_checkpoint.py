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
def _save_last_checkpoint(self, trainer: 'pl.Trainer', monitor_candidates: Dict[str, Tensor]) -> None:
    if not self.save_last:
        return
    filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)
    if self._enable_version_counter:
        version_cnt = self.STARTING_VERSION
        while self.file_exists(filepath, trainer) and filepath != self.last_model_path:
            filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST, ver=version_cnt)
            version_cnt += 1
    previous, self.last_model_path = (self.last_model_path, filepath)
    if self.save_last == 'link' and self._last_checkpoint_saved and (self.save_top_k != 0):
        self._link_checkpoint(trainer, self._last_checkpoint_saved, filepath)
    else:
        self._save_checkpoint(trainer, filepath)
    if previous and self._should_remove_checkpoint(trainer, previous, filepath):
        self._remove_checkpoint(trainer, previous)