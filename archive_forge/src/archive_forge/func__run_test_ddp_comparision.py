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
def _run_test_ddp_comparision(self, simulate_uneven_inputs=False):
    gLogger.info('Running trainer rank: %s', self.rank)
    torch.manual_seed(self.rank)
    dist.init_process_group(backend='gloo', init_method=INIT_METHOD_TEMPLATE.format(file_name=f'{self.file_name}_pg'), world_size=self.world_size, rank=self.rank)
    net = nn.Linear(2, 3)
    ddp_net = DistributedDataParallel(net)
    num_inputs = 1
    if simulate_uneven_inputs:
        if self.rank % 2 == 0:
            num_inputs += 2
    inputs_list = [torch.rand((3, 2)) for _ in range(num_inputs)]
    if simulate_uneven_inputs:
        gLogger.info('Rank %s training with %s inputs.', self.rank, len(inputs_list))
    grads_dict = {}
    with ddp_net.join(simulate_uneven_inputs):
        for i, inputs in enumerate(inputs_list):
            with dist_autograd.context() as context_id:
                loss = ddp_net(inputs).norm()
                dist_autograd.backward(context_id, [loss])
                grads_dict = dist_autograd.get_gradients(context_id)
            gLogger.info('Trainer #%s got grad dict: %s', self.rank, grads_dict)
            ddp_net.zero_grad()
            loss = ddp_net(inputs).norm()
            loss.backward()
            for param in net.parameters():
                self.assertTrue(param in grads_dict, msg=f'Param {param} is not in dist_auto grad dict {grads_dict} for iteration {i}')
                self.assertEqual(grads_dict[param], param.grad, msg=f'The grads for param {param} are different under local and dist autograd: {param.grad} \n---\n {grads_dict[param]} for iteration {i}')
    dist.destroy_process_group()