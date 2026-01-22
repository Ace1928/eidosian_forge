import sys
import threading
import time
from enum import Enum
import random
import torch
import torch.nn as nn
from datetime import timedelta
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_utils import IS_MACOS, skip_but_pass_in_sandcastle_if
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
class MyBackwardFunc(Function):

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    @once_differentiable
    def backward(ctx, input):
        assert DistAutogradTest._test_clean_context_backward_context_id is not None
        dist.barrier()
        dist_autograd._release_context(DistAutogradTest._test_clean_context_backward_context_id)
        assert _all_contexts_cleaned_up()
        return input