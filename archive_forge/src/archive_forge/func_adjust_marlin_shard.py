from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.utils import (
from vllm.model_executor.utils import set_weight_attrs
from vllm.logger import init_logger
def adjust_marlin_shard(param, shard_size, shard_offset):
    marlin_tile_size = getattr(param, 'marlin_tile_size', None)
    if marlin_tile_size is None:
        return (shard_size, shard_offset)
    return (shard_size * marlin_tile_size, shard_offset * marlin_tile_size)