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
def _monitor_candidates(self, trainer: 'pl.Trainer') -> Dict[str, Tensor]:
    monitor_candidates = deepcopy(trainer.callback_metrics)
    epoch = monitor_candidates.get('epoch')
    monitor_candidates['epoch'] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
    step = monitor_candidates.get('step')
    monitor_candidates['step'] = step.int() if isinstance(step, Tensor) else torch.tensor(trainer.global_step)
    return monitor_candidates