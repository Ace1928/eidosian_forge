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
def _create_default_config(self, zero_optimization: bool, zero_allow_untested_optimizer: bool, logging_batch_size_per_gpu: Optional[int], partition_activations: bool, cpu_checkpointing: bool, contiguous_memory_optimization: bool, synchronize_checkpoint_boundary: bool, offload_optimizer: bool, offload_parameters: bool, nvme_path: str, offload_params_device: str, params_buffer_count: int, params_buffer_size: int, max_in_cpu: int, offload_optimizer_device: str, optimizer_buffer_count: int, pin_memory: bool, block_size: int, queue_depth: int, single_submit: bool, overlap_events: bool, thread_count: int, **zero_kwargs: Any) -> Dict:
    cfg = {'activation_checkpointing': {'partition_activations': partition_activations, 'cpu_checkpointing': cpu_checkpointing, 'contiguous_memory_optimization': contiguous_memory_optimization, 'synchronize_checkpoint_boundary': synchronize_checkpoint_boundary}, 'aio': {'block_size': block_size, 'queue_depth': queue_depth, 'single_submit': single_submit, 'overlap_events': overlap_events, 'thread_count': thread_count}}
    if zero_optimization:
        zero_config = zero_kwargs
        if offload_optimizer:
            zero_config['offload_optimizer'] = {'device': offload_optimizer_device, 'nvme_path': nvme_path, 'buffer_count': optimizer_buffer_count, 'pin_memory': pin_memory}
        if offload_parameters:
            zero_config['offload_param'] = {'device': offload_params_device, 'nvme_path': nvme_path, 'buffer_count': params_buffer_count, 'buffer_size': params_buffer_size, 'max_in_cpu': max_in_cpu, 'pin_memory': pin_memory}
        cfg.update({'zero_allow_untested_optimizer': zero_allow_untested_optimizer, 'zero_optimization': zero_config})
    if logging_batch_size_per_gpu:
        cfg['train_micro_batch_size_per_gpu'] = logging_batch_size_per_gpu
    return cfg