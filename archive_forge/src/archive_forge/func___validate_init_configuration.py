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
def __validate_init_configuration(self) -> None:
    if self.save_top_k < -1:
        raise MisconfigurationException(f'Invalid value for save_top_k={self.save_top_k}. Must be >= -1')
    if self._every_n_train_steps < 0:
        raise MisconfigurationException(f'Invalid value for every_n_train_steps={self._every_n_train_steps}. Must be >= 0')
    if self._every_n_epochs < 0:
        raise MisconfigurationException(f'Invalid value for every_n_epochs={self._every_n_epochs}. Must be >= 0')
    every_n_train_steps_triggered = self._every_n_train_steps >= 1
    every_n_epochs_triggered = self._every_n_epochs >= 1
    train_time_interval_triggered = self._train_time_interval is not None
    if every_n_train_steps_triggered + every_n_epochs_triggered + train_time_interval_triggered > 1:
        raise MisconfigurationException(f'Combination of parameters every_n_train_steps={self._every_n_train_steps}, every_n_epochs={self._every_n_epochs} and train_time_interval={self._train_time_interval} should be mutually exclusive.')
    if self.monitor is None and self.save_top_k not in (-1, 0, 1):
        raise MisconfigurationException(f'ModelCheckpoint(save_top_k={self.save_top_k}, monitor=None) is not a valid configuration. No quantity for top_k to track.')