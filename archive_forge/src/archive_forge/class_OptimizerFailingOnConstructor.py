import threading
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.testing._internal.dist_utils import dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
class OptimizerFailingOnConstructor(optim.Optimizer):

    def __init__(self, params):
        super().__init__(params, {})
        raise ValueError('Error creating optimizer.')

    def step(self, closure=None):
        raise NotImplementedError