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
def _save_none_monitor_checkpoint(self, trainer: 'pl.Trainer', monitor_candidates: Dict[str, Tensor]) -> None:
    filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, self.best_model_path)
    previous, self.best_model_path = (self.best_model_path, filepath)
    self._save_checkpoint(trainer, filepath)
    if self.save_top_k == 1 and previous and self._should_remove_checkpoint(trainer, previous, filepath):
        self._remove_checkpoint(trainer, previous)