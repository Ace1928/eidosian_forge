from abc import ABC
from collections import defaultdict
from datetime import timedelta
import os
import torch
import torch.distributed as dist
from typing import Callable, List, T
import ray
from ray.actor import ActorHandle
from ray.train._internal.utils import get_address_and_port
from ray.train.constants import DEFAULT_NCCL_SOCKET_IFNAME
from ray.air._internal.torch_utils import get_device
def _init_torch_distributed(init_method: str, backend: str, rank: int, world_size: int, local_rank: int, local_world_size: int, master_addr: str, master_port: str, gpu_ids: List[int]):
    """Initialize torch distributed backend"""
    if init_method == 'env':
        os.environ['MASTER_ADDR'] = str(master_addr)
        os.environ['MASTER_PORT'] = str(master_port)
        url = 'env://'
    elif init_method == 'tcp':
        url = f'tcp://{master_addr}:{master_port}'
    else:
        raise ValueError(f"The provided init_method ({init_method}) is not supported. Must be either 'env' or 'tcp'.")
    if backend == 'nccl':
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join((str(gid) for gid in gpu_ids))
        if 'NCCL_SOCKET_IFNAME' not in os.environ:
            os.environ['NCCL_SOCKET_IFNAME'] = DEFAULT_NCCL_SOCKET_IFNAME
    dist.init_process_group(backend=backend, init_method=url, rank=rank, world_size=world_size, timeout=timedelta(seconds=1800))
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(local_world_size)