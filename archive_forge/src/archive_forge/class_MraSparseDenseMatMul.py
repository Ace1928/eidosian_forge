import math
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.cpp_extension import load
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mra import MraConfig
class MraSparseDenseMatMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sparse_query, indices, dense_key, query_num_block):
        sparse_qk_prod = sparse_dense_mm(sparse_query, indices, dense_key, query_num_block)
        ctx.save_for_backward(sparse_query, indices, dense_key)
        ctx.query_num_block = query_num_block
        return sparse_qk_prod

    @staticmethod
    def backward(ctx, grad):
        sparse_query, indices, dense_key = ctx.saved_tensors
        query_num_block = ctx.query_num_block
        key_num_block = dense_key.size(1) // sparse_query.size(-1)
        indices_T = transpose_indices(indices, query_num_block, key_num_block)
        grad_key = sparse_dense_mm(sparse_query.transpose(-1, -2), indices_T, grad, key_num_block)
        grad_query = mm_to_sparse(grad, dense_key, indices)
        return (grad_query, None, grad_key, None)

    @staticmethod
    def operator_call(sparse_query, indices, dense_key, query_num_block):
        return MraSparseDenseMatMul.apply(sparse_query, indices, dense_key, query_num_block)