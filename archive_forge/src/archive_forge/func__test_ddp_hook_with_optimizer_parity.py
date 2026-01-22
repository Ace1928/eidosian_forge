import copy
import itertools
import math
import os
import random
import sys
import tempfile
import time
from collections import namedtuple, OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from functools import reduce
from typing import Union, NamedTuple, Callable, Any
import unittest
import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.model_averaging.hierarchical_model_averager as hierarchicalSGD
import torch.distributed.algorithms.model_averaging.utils as model_averaging_utils
import torch.nn as nn
import torch.nn.functional as F
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.algorithms.ddp_comm_hooks import (
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.distributed.distributed_c10d import (
from torch.distributed.utils import (
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.distributed import _dump_DDP_relevant_env_vars, _MixedPrecision
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import (
import torch.distributed.optim.post_localSGD_optimizer as post_localSGD_optimizer
from torch.utils.data.distributed import DistributedSampler
def _test_ddp_hook_with_optimizer_parity(self, grad_as_bucket_view, static_graph, optim_cls, optimize_subset, *functional_optim_args, **functional_optim_kwargs):
    rank = self.rank
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    models_to_test = [(LargeNet(), torch.randn(1, 1000).cuda())]
    if HAS_TORCHVISION:
        models_to_test.append((torchvision.models.resnet50(), torch.randn(1, 3, 3, 1000).cuda()))
    for model, inp in models_to_test:
        with torch.backends.cudnn.flags(enabled=True, deterministic=True, benchmark=False):
            ddp_model_with_optimizer_hook = torch.nn.parallel.DistributedDataParallel(copy.deepcopy(model).cuda(), device_ids=[self.rank], gradient_as_bucket_view=grad_as_bucket_view, static_graph=static_graph)
            ddp_model_with_no_hook = torch.nn.parallel.DistributedDataParallel(copy.deepcopy(model).cuda(), device_ids=[self.rank], gradient_as_bucket_view=grad_as_bucket_view, static_graph=static_graph)
            hook_params = ddp_model_with_optimizer_hook.parameters()
            no_hook_params = ddp_model_with_no_hook.parameters()
            if optimize_subset:
                hook_params = list(hook_params)
                no_hook_params = list(no_hook_params)
                self.assertGreater(len(hook_params), 0)
                hook_params = [hook_params[0]]
                no_hook_params = [no_hook_params[0]]
            if optimize_subset:
                ddp_model_with_optimizer_hook._register_fused_optim(optim_cls, *functional_optim_args, optim_params=hook_params, **functional_optim_kwargs)
            else:
                ddp_model_with_optimizer_hook._register_fused_optim(optim_cls, *functional_optim_args, **functional_optim_kwargs)
            optimizer_no_hook = optim_cls(no_hook_params, *functional_optim_args, **functional_optim_kwargs)
            for hook_param, allreduce_param in zip(ddp_model_with_optimizer_hook.parameters(), ddp_model_with_no_hook.parameters()):
                self.assertEqual(hook_param, allreduce_param)
            opt_hook_init_params = copy.deepcopy(list(ddp_model_with_optimizer_hook.parameters()))
            for i in range(6):
                ddp_model_with_optimizer_hook.zero_grad()
                out = ddp_model_with_optimizer_hook(inp)
                loss = out.sum()
                loss.backward()
            dist.barrier()
            for i in range(6):
                ddp_model_with_no_hook.zero_grad()
                out = ddp_model_with_no_hook(inp)
                loss = out.sum()
                loss.backward()
                optimizer_no_hook.step()
            dist.barrier()
            for hook_param, allreduce_param in zip(ddp_model_with_optimizer_hook.parameters(), ddp_model_with_no_hook.parameters()):
                self.assertEqual(hook_param, allreduce_param)
            if optimize_subset:
                self.assertNotEqual(opt_hook_init_params[0], next(iter(ddp_model_with_optimizer_hook.parameters())))
                self.assertEqual(opt_hook_init_params[1:], list(ddp_model_with_optimizer_hook.parameters())[1:])
            else:
                self.assertNotEqual(opt_hook_init_params, list(ddp_model_with_optimizer_hook.parameters()))
            dist.barrier()