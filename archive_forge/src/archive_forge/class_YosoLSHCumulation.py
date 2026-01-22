import math
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_yoso import YosoConfig
class YosoLSHCumulation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, query_mask, key_mask, query, key, value, config):
        if query_mask.size(0) != key_mask.size(0):
            raise ValueError('Query mask and Key mask differ in sizes in dimension 0')
        if query_mask.size(0) != query.size(0):
            raise ValueError('Query mask and Query differ in sizes in dimension 0')
        if query_mask.size(0) != key.size(0):
            raise ValueError('Query mask and Key differ in sizes in dimension 0')
        if query_mask.size(0) != value.size(0):
            raise ValueError('Query mask and Value mask differ in sizes in dimension 0')
        if key.size(1) != value.size(1):
            raise ValueError('Key and Value differ in sizes in dimension 1')
        if query.size(2) != key.size(2):
            raise ValueError('Query and Key differ in sizes in dimension 2')
        query_mask, key_mask, query, key, value = to_contiguous([query_mask, key_mask, query, key, value])
        use_cuda = query_mask.is_cuda
        num_hash = config['num_hash']
        hash_code_len = config['hash_code_len']
        hashtable_capacity = int(2 ** hash_code_len)
        if config['use_fast_hash']:
            query_hash_code, key_hash_code = lsh_cumulation.fast_hash(query_mask, query, key_mask, key, num_hash, hash_code_len, use_cuda, 1)
        else:
            query_hash_code, key_hash_code = hashing(query, key, num_hash, hash_code_len)
        cumulation_value = lsh_cumulation.lsh_cumulation(query_mask, query_hash_code, key_mask, key_hash_code, value, hashtable_capacity, use_cuda, 1)
        ctx.save_for_backward(query_mask, key_mask, query_hash_code, key_hash_code, query, key, value)
        ctx.config = config
        return cumulation_value

    @staticmethod
    def backward(ctx, grad):
        grad = to_contiguous(grad)
        query_mask, key_mask, query_hash_code, key_hash_code, query, key, value = ctx.saved_tensors
        config = ctx.config
        use_cuda = grad.is_cuda
        hash_code_len = config['hash_code_len']
        hashtable_capacity = int(2 ** hash_code_len)
        if config['lsh_backward']:
            grad_value = lsh_cumulation.lsh_cumulation(key_mask, key_hash_code, query_mask, query_hash_code, grad, hashtable_capacity, use_cuda, 1)
            grad_query = lsh_cumulation.lsh_weighted_cumulation(query_mask, query_hash_code, grad, key_mask, key_hash_code, value, hash_code_len / 2 * key, hashtable_capacity, use_cuda, 4)
            grad_key = lsh_cumulation.lsh_weighted_cumulation(key_mask, key_hash_code, value, query_mask, query_hash_code, grad, hash_code_len / 2 * query, hashtable_capacity, use_cuda, 4)
        else:
            expectation = (1 - torch.acos(torch.matmul(query, key.transpose(-1, -2))) / math.pi) ** hash_code_len
            expectation = expectation * query_mask[:, :, None] * key_mask[:, None, :]
            weighted_exp = torch.matmul(grad, value.transpose(-1, -2)) * expectation
            grad_query = torch.matmul(weighted_exp, hash_code_len / 2 * key)
            grad_key = torch.matmul(weighted_exp.transpose(-1, -2), hash_code_len / 2 * query)
            grad_value = torch.matmul(expectation.transpose(-1, -2), grad)
        return (None, None, grad_query, grad_key, grad_value, None)