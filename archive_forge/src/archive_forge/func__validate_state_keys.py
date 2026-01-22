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
def _validate_state_keys(state: Dict[str, Any]) -> None:
    deepspeed_internal_keys = {'module', 'buffer_names', 'optimizer', 'param_shapes', 'lr_scheduler', 'sparse_tensor_module_names', 'skipped_steps', 'global_steps', 'global_samples', 'dp_world_size', 'mp_world_size', 'ds_config', 'ds_version'}
    colliding_keys = deepspeed_internal_keys.intersection(state.keys())
    if colliding_keys:
        rank_zero_warn("Your state has keys that collide with DeepSpeed's internal engine state. This could result in your values being overwritten by DeepSpeed. Consider changing the name of these keys to something else: " + ', '.join(colliding_keys))