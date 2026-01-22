import copy
import functools
import inspect
import itertools
import logging
import os
import sys
import warnings
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import auto, Enum
from typing import Any, Callable, List, Optional, Type
import torch
import torch.distributed as dist
from torch.autograd import Function, Variable
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch._utils import _get_device_index
from ..modules import Module
from .scatter_gather import gather, scatter_kwargs  # noqa: F401
def _dump_DDP_relevant_env_vars():
    relevant_env_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_PORT', 'MASTER_ADDR', 'CUDA_VISIBLE_DEVICES', 'GLOO_SOCKET_IFNAME', 'GLOO_DEVICE_TRANSPORT', 'NCCL_SOCKET_IFNAME', 'TORCH_NCCL_BLOCKING_WAIT', 'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS', 'NCCL_IB_DISABLE', 'NCCL_P2P_DISABLE', 'NCCL_P2P_LEVEL', 'NCCL_SHM_DISABLE', 'NCCL_SOCKET_NTHREADS', 'NCCL_NSOCKS_PERTHREAD', 'NCCL_BUFFSIZE', 'NCCL_NTHREADS', 'NCCL_RINGS', 'NCCL_MAX_NCHANNELS', 'NCCL_MIN_NCHANNELS', 'NCCL_CHECKS_DISABLE', 'NCCL_CHECK_POINTERS', 'NCCL_LAUNCH_MODE', 'NCCL_IB_HCA', 'NCCL_IB_TIMEOUT', 'NCCL_IB_RETRY_CNT', 'NCCL_IB_GID_INDEX', 'NCCL_IB_SL', 'NCCL_IB_TC', 'NCCL_IB_AR_THRESHOLD', 'NCCL_IB_CUDA_SUPPORT', 'NCCL_NET_GDR_LEVEL', 'NCCL_NET_GDR_READ', 'NCCL_SINGLE_RING_THRESHOLD', 'NCCL_LL_THRESHOLD', 'NCCL_TREE_THRESHOLD', 'NCCL_ALGO', 'NCCL_PROTO', 'NCCL_IGNORE_CPU_AFFINITY', 'NCCL_DEBUG_FILE', 'NCCL_COLLNET_ENABLE', 'NCCL_TOPO_FILE', 'NCCL_TOPO_DUMP_FILE', 'TORCH_NCCL_ASYNC_ERROR_HANDLING']
    formatted_output = ''
    for var in relevant_env_vars:
        value = os.environ[var] if var in os.environ else 'N/A'
        formatted_output += f'env:{var}={value}\n'
    print(formatted_output)