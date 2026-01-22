import shutil
from contextlib import ExitStack
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import (
import torch
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from typing_extensions import TypeGuard, override
from lightning_fabric.accelerators import Accelerator
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment, Precision
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.plugins.precision.fsdp import FSDPPrecision
from lightning_fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning_fabric.strategies.parallel import ParallelStrategy
from lightning_fabric.strategies.registry import _StrategyRegistry
from lightning_fabric.strategies.strategy import (
from lightning_fabric.utilities.distributed import (
from lightning_fabric.utilities.distributed import group as _group
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.init import _EmptyInit
from lightning_fabric.utilities.load import _METADATA_FILENAME, _lazy_load, _materialize_tensors, _move_state_into
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_only, rank_zero_warn
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH, _Stateful
def _activation_checkpointing_kwargs(activation_checkpointing: Optional[Union[Type[Module], List[Type[Module]]]], activation_checkpointing_policy: Optional['_POLICY']) -> Dict:
    if activation_checkpointing is None and activation_checkpointing_policy is None:
        return {}
    if activation_checkpointing is not None and activation_checkpointing_policy is not None:
        raise ValueError('You cannot set both `activation_checkpointing` and `activation_checkpointing_policy`. Use the latter.')
    if activation_checkpointing is not None:
        if isinstance(activation_checkpointing, list):
            classes = tuple(activation_checkpointing)
        else:
            classes = (activation_checkpointing,)
        if _TORCH_GREATER_EQUAL_2_1:
            rank_zero_deprecation(f'`FSDPStrategy(activation_checkpointing={activation_checkpointing})` is deprecated, use `FSDPStrategy(activation_checkpointing_policy={set(classes)})` instead.')
        return {'check_fn': lambda submodule: isinstance(submodule, classes)}
    if isinstance(activation_checkpointing_policy, set):
        if _TORCH_GREATER_EQUAL_2_1:
            return _auto_wrap_policy_kwargs(activation_checkpointing_policy, {})
        return {'check_fn': lambda submodule: isinstance(submodule, tuple(activation_checkpointing_policy))}
    if not _TORCH_GREATER_EQUAL_2_1:
        raise ValueError('`activation_checkpointing_policy` requires torch >= 2.1.0. HINT: `pip install -U torch`')
    return {'auto_wrap_policy': activation_checkpointing_policy}