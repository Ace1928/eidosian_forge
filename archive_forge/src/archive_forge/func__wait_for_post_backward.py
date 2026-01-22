import contextlib
import copy
from enum import Enum, auto
import functools
import logging
from math import inf
import os
import time
import traceback
import typing
from typing import (
import torch
from torch.autograd import Variable
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fairscale.internal.containers import apply_to_tensors
from fairscale.internal.parallel import (
from fairscale.internal.params import calc_grad_norm, recursive_copy_to_device
from fairscale.internal.reduce_scatter_bucketer import ReduceScatterBucketer
from fairscale.internal.state_dict import replace_by_prefix_
from fairscale.nn.misc import FlattenParamsWrapper, _enable_pre_load_state_dict_hook
from fairscale.nn.wrap import auto_wrap, config_auto_wrap_policy, enable_wrap
from . import fsdp_optim_utils as ou
@torch.no_grad()
def _wait_for_post_backward(self) -> None:
    """Wait for post-backward to finish. Only called on root instance."""
    p_assert(self._is_root, 'WFPB not called on root')
    if any([p.requires_grad for p in self.params]):
        self.assert_state(TrainingState.BACKWARD_POST)
    else:
        self.assert_state(TrainingState.BACKWARD_PRE)
    if self._require_backward_grad_sync:
        with torch.cuda.stream(self._streams['post_backward']):
            p_assert(self._reducer is not None, 'WFPB: reducer is None')
            assert self._reducer is not None
            self._reducer.flush()
        torch.cuda.current_stream().wait_stream(self._streams['post_backward'])
        if self.move_grads_to_cpu:
            torch.cuda.current_stream().synchronize()
    if self._reducer is not None:
        self._reducer.teardown()

    def _finalize_parameters(fsdp_module: FullyShardedDataParallel) -> None:
        """Helper used below on all fsdp modules."""
        if not fsdp_module._is_root and self._require_backward_grad_sync:
            fsdp_module._free_full_params()
            fsdp_module._use_fp32_param_shard()
        for p in fsdp_module.params:
            if not p.requires_grad:
                continue
            if not self._require_backward_grad_sync:
                continue
            if hasattr(p, '_cpu_grad'):
                p_assert(p.device == torch.device('cpu'), f'WFPB: incorrect cpu_grad device {p.device}')
                p.grad = p._cpu_grad
            elif hasattr(p, '_saved_grad_shard'):
                p_assert(p.device == p._saved_grad_shard.device, f'WFPB: incorrect saved_grad_shard device p.device={p.device} vs p._saved_grad_shard.device={p._saved_grad_shard.device}')
                p_assert(p.shape == p._saved_grad_shard.shape, f'WFPB: incorrect saved_grad_shard shape p.shape={p.shape} vs p._saved_grad_shard.shape={p._saved_grad_shard.shape}')
                p.grad = p._saved_grad_shard
            if hasattr(p, '_saved_grad_shard'):
                delattr(p, '_saved_grad_shard')
    for m in get_fsdp_instances(self):
        _finalize_parameters(m)
        m._pre_backward_hook_has_run = False
        if any((p.requires_grad for p in m.parameters())):
            if any([p.requires_grad for p in m.params]):
                m.assert_state(TrainingState.BACKWARD_POST)
            else:
                m.assert_state(TrainingState.BACKWARD_PRE)
        else:
            m.assert_state([TrainingState.BACKWARD_PRE, TrainingState.IDLE])
        m.training_state = TrainingState.IDLE
        if m._is_root:
            self._post_backward_callback_queued = False
            p_assert(self._output_pre_backward_hook_registered is not None, 'WFPB: self._output_pre_backward_hook_registered should not be None')
            assert self._output_pre_backward_hook_registered is not None
            self._output_pre_backward_hook_registered.clear()