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
def _get_node_and_gpu_ids():
    """Returns the node_id and gpu_ids for this worker."""
    node_id = ray.get_runtime_context().get_node_id()
    gpu_ids = ray.get_gpu_ids()
    return (node_id, gpu_ids)