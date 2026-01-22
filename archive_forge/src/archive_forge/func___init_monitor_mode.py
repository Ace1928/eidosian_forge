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
def __init_monitor_mode(self, mode: str) -> None:
    torch_inf = torch.tensor(torch.inf)
    mode_dict = {'min': (torch_inf, 'min'), 'max': (-torch_inf, 'max')}
    if mode not in mode_dict:
        raise MisconfigurationException(f'`mode` can be {', '.join(mode_dict.keys())} but got {mode}')
    self.kth_value, self.mode = mode_dict[mode]