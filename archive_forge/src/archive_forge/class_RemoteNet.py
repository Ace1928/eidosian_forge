import contextlib
import enum
import logging
import os
import threading
from typing import NamedTuple
import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.nn as nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
from torch.testing._internal.dist_utils import INIT_METHOD_TEMPLATE, dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
class RemoteNet(nn.Module):

    def __init__(self, d_in: int, d_out: int):
        gLogger.info('Initing RemoteNet with %s %s', d_in, d_out)
        super().__init__()
        self.fc = getLinear(d_in, d_out)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        gLogger.debug('Running RemoteNet.forward() on: %s', input)
        return self.relu(self.fc(input))