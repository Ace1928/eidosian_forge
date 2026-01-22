import warnings
import sys
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import Tuple, Union, List, Optional, cast, TYPE_CHECKING
from . import _functional_collectives_impl as fun_col_impl
from ._functional_collectives_impl import _register_tensor_wrapper
from torch.fx.experimental.proxy_tensor import (
from torch._custom_ops import impl_abstract
from torch.distributed.distributed_c10d import (
def _register_ops():
    ops_defs = ['broadcast(Tensor self, int src, str tag, int[] ranks, int group_size) -> Tensor', 'all_reduce(Tensor self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor', 'all_reduce_coalesced(Tensor[] self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]', 'wait_tensor(Tensor self) -> Tensor', 'all_gather_into_tensor(Tensor shard, str tag, int[] ranks, int group_size) -> Tensor', 'all_gather_into_tensor_coalesced(Tensor[] input, str tag, int[] ranks, int group_size) -> Tensor[]', 'reduce_scatter_tensor(Tensor input, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor', 'reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]', 'all_to_all_single(Tensor input, SymInt[]? output_split_sizes, SymInt[]? input_split_sizes, str tag, int[] ranks, int group_size) -> Tensor']
    my_module = sys.modules[__name__]
    for op_def in ops_defs:
        op_name = op_def[0:op_def.index('(')]
        backend_impl = getattr(fun_col_impl, f'_{op_name}')
        meta_impl = getattr(my_module, f'_{op_name}_meta')
        c10_lib.define(op_def, tags=torch.Tag.pt2_compliant_tag)
        c10_lib_impl.impl(op_name, backend_impl, 'CompositeExplicitAutograd')
        impl_abstract(f'c10d_functional::{op_name}')(meta_impl)