import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional
import torch
import torch.distributed as dist
import ray
from ray.train._internal.utils import get_address_and_port
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import Backend, BackendConfig
from ray.train.constants import DEFAULT_NCCL_SOCKET_IFNAME
from ray.util import PublicAPI
def _set_torch_distributed_env_vars():
    from ray.air._internal.torch_utils import get_device
    context = ray.train.get_context()
    os.environ['LOCAL_RANK'] = str(context.get_local_rank())
    os.environ['RANK'] = str(context.get_world_rank())
    os.environ['LOCAL_WORLD_SIZE'] = str(context.get_local_world_size())
    os.environ['WORLD_SIZE'] = str(context.get_world_size())
    os.environ['NODE_RANK'] = str(context.get_node_rank())
    device = get_device()
    if isinstance(device, list):
        device = device[0]
    os.environ['ACCELERATE_TORCH_DEVICE'] = str(device)