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
def _set_parallel_devices_and_init_accelerator(self) -> None:
    if isinstance(self._accelerator_flag, Accelerator):
        self.accelerator: Accelerator = self._accelerator_flag
    else:
        self.accelerator = AcceleratorRegistry.get(self._accelerator_flag)
    accelerator_cls = self.accelerator.__class__
    if not accelerator_cls.is_available():
        available_accelerator = [acc_str for acc_str in self._accelerator_types if AcceleratorRegistry[acc_str]['accelerator'].is_available()]
        raise MisconfigurationException(f'`{accelerator_cls.__qualname__}` can not run on your system since the accelerator is not available. The following accelerator(s) is available and can be passed into `accelerator` argument of `Trainer`: {available_accelerator}.')
    self._set_devices_flag_if_auto_passed()
    self._devices_flag = accelerator_cls.parse_devices(self._devices_flag)
    if not self._parallel_devices:
        self._parallel_devices = accelerator_cls.get_parallel_devices(self._devices_flag)