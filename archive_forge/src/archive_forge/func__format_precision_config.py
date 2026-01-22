import argparse
import json
import logging
import os
import platform
from contextlib import ExitStack
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, List, Mapping, Optional, Tuple, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override
from lightning_fabric.accelerators import Accelerator, CUDAAccelerator
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies.ddp import DDPStrategy
from lightning_fabric.strategies.registry import _StrategyRegistry
from lightning_fabric.strategies.strategy import _Sharded
from lightning_fabric.utilities.distributed import log
from lightning_fabric.utilities.load import _move_state_into
from lightning_fabric.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH
def _format_precision_config(config: Dict[str, Any], precision: str, loss_scale: float, loss_scale_window: int, min_loss_scale: int, initial_scale_power: int, hysteresis: int) -> None:
    if 'fp16' not in config and precision in ('16-mixed', '16-true'):
        rank_zero_info('Enabling DeepSpeed FP16. Model parameters and inputs will be cast to `float16`.')
        config['fp16'] = {'enabled': True, 'loss_scale': loss_scale, 'initial_scale_power': initial_scale_power, 'loss_scale_window': loss_scale_window, 'hysteresis': hysteresis, 'min_loss_scale': min_loss_scale}
    elif 'bf16' not in config and precision in ('bf16-mixed', 'bf16-true'):
        rank_zero_info('Enabling DeepSpeed BF16. Model parameters and inputs will be cast to `bfloat16`.')
        config['bf16'] = {'enabled': True}