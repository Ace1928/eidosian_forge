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
def _check_and_init_precision(self) -> Precision:
    self._validate_precision_choice()
    if isinstance(self._precision_plugin_flag, Precision):
        return self._precision_plugin_flag
    if _graphcore_available_and_importable():
        from lightning_graphcore import IPUAccelerator, IPUPrecision
        if isinstance(self.accelerator, IPUAccelerator):
            return IPUPrecision(self._precision_flag)
    if _habana_available_and_importable():
        from lightning_habana import HPUAccelerator, HPUPrecisionPlugin
        if isinstance(self.accelerator, HPUAccelerator):
            return HPUPrecisionPlugin(self._precision_flag)
    if _LIGHTNING_COLOSSALAI_AVAILABLE:
        from lightning_colossalai import ColossalAIPrecisionPlugin, ColossalAIStrategy
        if isinstance(self.strategy, ColossalAIStrategy):
            return ColossalAIPrecisionPlugin(self._precision_flag)
    if isinstance(self.strategy, (SingleDeviceXLAStrategy, XLAStrategy)):
        return XLAPrecision(self._precision_flag)
    if isinstance(self.strategy, DeepSpeedStrategy):
        return DeepSpeedPrecision(self._precision_flag)
    if isinstance(self.strategy, FSDPStrategy):
        return FSDPPrecision(self._precision_flag)
    if self._precision_flag in ('16-true', 'bf16-true'):
        return HalfPrecision(self._precision_flag)
    if self._precision_flag == '32-true':
        return Precision()
    if self._precision_flag == '64-true':
        return DoublePrecision()
    if self._precision_flag == 'transformer-engine':
        return TransformerEnginePrecision(weights_dtype=torch.bfloat16)
    if self._precision_flag == 'transformer-engine-float16':
        return TransformerEnginePrecision(weights_dtype=torch.float16)
    if self._precision_flag == '16-mixed' and self._accelerator_flag == 'cpu':
        rank_zero_warn("You passed `Trainer(accelerator='cpu', precision='16-mixed')` but AMP with fp16 is not supported on CPU. Using `precision='bf16-mixed'` instead.")
        self._precision_flag = 'bf16-mixed'
    if self._precision_flag in ('16-mixed', 'bf16-mixed'):
        rank_zero_info(f'Using {('16bit' if self._precision_flag == '16-mixed' else 'bfloat16')} Automatic Mixed Precision (AMP)')
        device = 'cpu' if self._accelerator_flag == 'cpu' else 'cuda'
        return MixedPrecision(self._precision_flag, device)
    raise RuntimeError('No precision set')