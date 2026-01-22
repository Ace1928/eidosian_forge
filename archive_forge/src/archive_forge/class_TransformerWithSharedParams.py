import itertools
import os
import re
import sys
from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import deepcopy
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from unittest import mock
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp._init_utils import NO_RESHARD_AFTER_FORWARD_STRATEGIES
from torch.distributed.fsdp.fully_sharded_data_parallel import (
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy, wrap
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import FILE_SCHEMA, get_cycles_per_ms
class TransformerWithSharedParams(FSDPTestModel):

    def __init__(self, group: dist.ProcessGroup, cuda_init_mode: CUDAInitMode, add_bn: bool, deterministic: bool):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        if deterministic:
            torch.manual_seed(0)
        d_vocab = 23
        d_model = 16
        self.embed_tokens = nn.Embedding(d_vocab, d_model)
        self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=8, dropout=0.1)
        self.output_proj = nn.Linear(d_model, d_vocab)
        self.output_proj.weight = self.embed_tokens.weight
        self.register_buffer('vocab_bias', self.embed_tokens.weight.new_ones((d_model,)))
        self.register_buffer('long_buffer', torch.zeros_like(self.vocab_bias, dtype=torch.long))
        self.bs = 2
        self.bn = torch.nn.BatchNorm1d(self.bs) if add_bn else torch.nn.Identity()
        if cuda_init_mode == CUDAInitMode.CUDA_BEFORE:
            self = self.cuda()
        if deterministic:
            self.eval()

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)
        src = torch.arange(12, device=device).view(6, self.bs)
        tgt = torch.arange(self.bs * 4, device=device).view(4, self.bs)
        return (src, tgt)

    def forward(self, src_ids, tgt_ids):
        src = self.embed_tokens(src_ids)
        src = src + self.vocab_bias + self.long_buffer.type_as(src)
        tgt = self.embed_tokens(tgt_ids)
        tgt = self.bn(tgt)
        x = self.transformer(src, tgt)
        return self.output_proj(x)

    def get_loss(self, input, output):
        _, tgt = input
        return nn.functional.cross_entropy(output.view(-1, output.size(-1)), tgt.view(-1), reduction='sum')

    def run_backward(self, loss):
        loss.backward()

    @staticmethod
    def init(group: dist.ProcessGroup, fsdp_init_mode: FSDPInitMode, cuda_init_mode: CUDAInitMode, fsdp_kwargs: Optional[Dict[str, Any]]=None, deterministic: bool=False, add_bn: bool=True) -> Union[nn.Module, FSDP]:
        """
        Initializes a :class:`TransformerWithSharedParams` instance.

        Args:
            fsdp_init_mode (FSDPInitMode): If ``NO_FSDP``, then does not wrap
                any modules with FSDP. If ``RECURSIVE``, then wraps with
                top-level FSDP. By default, the top-level FSDP uses the
                ``ModuleWrapPolicy`` for encoder and decoder layers, but a
                different auto wrap policy may be specified via
                ``fsdp_kwargs``.
            cuda_init_mode (CUDAInitMode): Determines model movement to CUDA.
            fsdp_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments
                forwarded to the FSDP constructor.
            deterministic (bool): Whether to make the model deterministic
                across constructions.
            add_bn (bool): Whether to include batch norm in the model.
        """
        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            if isinstance(group, tuple):
                pg = group[0]
            else:
                pg = group
            return TransformerWithSharedParams(pg, cuda_init_mode, add_bn, deterministic)
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            if 'auto_wrap_policy' not in fsdp_kwargs:
                auto_wrap_policy = ModuleWrapPolicy({TransformerEncoderLayer, TransformerDecoderLayer})
            else:
                auto_wrap_policy = fsdp_kwargs.pop('auto_wrap_policy')
            if 'sharding_strategy' in fsdp_kwargs and fsdp_kwargs['sharding_strategy'] in {ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2} and (not isinstance(group, tuple)):
                fsdp_pg = None
            else:
                fsdp_pg = group
            if isinstance(group, tuple):
                tformer_pg = group[0]
            else:
                tformer_pg = group
            m = TransformerWithSharedParams(tformer_pg, cuda_init_mode, add_bn, deterministic)
            fsdp_model = FSDP(m, fsdp_pg, auto_wrap_policy=auto_wrap_policy, **fsdp_kwargs)
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            return fsdp_model
        raise ValueError(f'Unsupported FSDP init mode: {fsdp_init_mode}')

    def get_ignored_modules(self):
        return [self.transformer]