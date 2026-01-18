from typing import Optional, Sequence
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.utils import divide
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.utils import set_weight_attrs
def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int) -> Sequence[int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank)