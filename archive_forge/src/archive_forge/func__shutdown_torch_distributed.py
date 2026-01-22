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
def _shutdown_torch_distributed():
    """Shutdown torch distributed backend"""
    dist.destroy_process_group()
    if not torch.cuda.is_available():
        return
    devices = get_device()
    if not isinstance(devices, list):
        devices = [devices]
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()