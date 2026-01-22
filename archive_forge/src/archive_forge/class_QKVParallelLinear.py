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
class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        linear_method: (Maybe quantized) linear method.
    """

    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int, total_num_kv_heads: Optional[int]=None, bias: bool=True, skip_bias_add: bool=False, params_dtype: Optional[torch.dtype]=None, linear_method: Optional[LinearMethodBase]=None):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        super().__init__(input_size, output_size, bias, False, skip_bias_add, params_dtype, linear_method)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: Optional[str]=None):
        param_data = param.data
        output_dim = getattr(param, 'output_dim', None)
        if loaded_shard_id is None:
            if output_dim is None:
                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offsets = [('q', 0, self.total_num_heads * self.head_size), ('k', self.total_num_heads * self.head_size, self.total_num_kv_heads * self.head_size), ('v', (self.total_num_heads + self.total_num_kv_heads) * self.head_size, self.total_num_kv_heads * self.head_size)]
            packed_dim = getattr(param, 'packed_dim', None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                    shard_size, shard_offset = adjust_marlin_shard(param, shard_size, shard_offset)
                loaded_weight_shard = loaded_weight.narrow(output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return
        tp_rank = get_tensor_model_parallel_rank()
        assert loaded_shard_id in ['q', 'k', 'v']
        if output_dim is not None:
            if loaded_shard_id == 'q':
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == 'k':
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == 'v':
                shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            packed_dim = getattr(param, 'packed_dim', None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
                shard_size, shard_offset = adjust_marlin_shard(param, shard_size, shard_offset)
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            if loaded_shard_id == 'q':
                shard_id = tp_rank
            else:
                shard_id = tp_rank // self.num_kv_head_replicas
            start_idx = shard_id * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        else:
            ignore_warning = getattr(param, 'ignore_warning', False)
            if not ignore_warning:
                logger.warning('Loading a weight without `output_dim` attribute in QKVParallelLinear, assume the weight is the same for all partitions.')
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)