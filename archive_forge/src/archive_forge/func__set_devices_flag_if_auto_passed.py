import logging
import os
from collections import Counter
from typing import Dict, List, Literal, Optional, Union
import torch
from lightning_fabric.connector import _PRECISION_INPUT, _PRECISION_INPUT_STR, _convert_precision_to_unified_args
from lightning_fabric.plugins.environments import (
from lightning_fabric.utilities.device_parser import _determine_root_gpu_device
from lightning_fabric.utilities.imports import _IS_INTERACTIVE
from pytorch_lightning.accelerators import AcceleratorRegistry
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator
from pytorch_lightning.accelerators.xla import XLAAccelerator
from pytorch_lightning.plugins import (
from pytorch_lightning.plugins.layer_sync import LayerSync, TorchSyncBatchNorm
from pytorch_lightning.strategies import (
from pytorch_lightning.strategies.ddp import _DDP_FORK_ALIASES
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import (
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _set_devices_flag_if_auto_passed(self) -> None:
    if self._devices_flag != 'auto':
        return
    if _IS_INTERACTIVE and isinstance(self.accelerator, CUDAAccelerator) and (self.accelerator.auto_device_count() > 1):
        self._devices_flag = 1
        rank_zero_info(f'Trainer will use only 1 of {self.accelerator.auto_device_count()} GPUs because it is running inside an interactive / notebook environment. You may try to set `Trainer(devices={self.accelerator.auto_device_count()})` but please note that multi-GPU inside interactive / notebook environments is considered experimental and unstable. Your mileage may vary.')
    else:
        self._devices_flag = self.accelerator.auto_device_count()