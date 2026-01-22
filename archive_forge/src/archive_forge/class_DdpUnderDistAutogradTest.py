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
class DdpUnderDistAutogradTest(RpcAgentTestFixture):

    @property
    def world_size(self) -> int:
        return WORLD_SIZE

    def remote_worker_name(self) -> str:
        return f'worker{REMOTE_WORKER_RANK}'

    def trainer_name(self, rank):
        return f'worker{rank}'

    def _remote_worker_process(self, ddp_mode):
        gLogger.info('The remote worker is running.')
        dist.init_process_group(backend='gloo', init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name), world_size=self.world_size, rank=self.rank)
        if ddp_mode in (DdpMode.INSIDE, DdpMode.OUTSIDE):
            dist.new_group(TRAINER_RANKS)
        global shutdown_signal
        with shutdown_signal:
            shutdown_signal.wait()
        gLogger.info('Exiting remote worker.')
        dist.destroy_process_group()

    def _trainer_process(self, rank: int):
        gLogger.info('Running the trainer #%s...', rank)
        gLogger.info('Initing trainer process group by trainer #%s with ranks %s', rank, TRAINER_RANKS)
        dist.init_process_group(backend='gloo', init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name), world_size=self.world_size, rank=self.rank)
        gLogger.info('Waiting for shutdown signal on trainer #%s...', rank)
        global shutdown_signal
        with shutdown_signal:
            shutdown_signal.wait()
        gLogger.info('Exiting the trainer #%s...', rank)
        dist.destroy_process_group()

    def _master_process(self, ddp_mode: DdpMode, simulate_uneven_inputs: bool):
        gLogger.info('Running the master process...')
        dist.init_process_group(backend='gloo', init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name), world_size=self.world_size, rank=self.rank)
        remote_em_rref = rpc.remote(self.remote_worker_name(), RemoteEM, args=(NUM_EM_ROW, D_SPARSE))
        remote_net_rref = rpc.remote(self.remote_worker_name(), RemoteNet, args=(D_DENSE + D_SPARSE, D_HID))
        gLogger.info('Created remote rrefs on master')
        self.do_test_on_master(ddp_mode, simulate_uneven_inputs, remote_em_rref, remote_net_rref)

    def do_test_on_master(self, ddp_mode: DdpMode, simulate_uneven_inputs: bool, remote_em_rref: rpc.RRef, remote_net_rref: rpc.RRef):
        if simulate_uneven_inputs:
            gLogger.info('Running DDP + RPC test with simulating uneven inputs across trainers.')
        trainer_rrefs = []
        for rank in TRAINER_RANKS:
            trainer = self.trainer_name(rank)
            trainer_rrefs.append(rpc.remote(trainer, Trainer, args=(remote_em_rref, remote_net_rref, ddp_mode, rank)))
        if ddp_mode in (DdpMode.INSIDE, DdpMode.OUTSIDE):
            dist.new_group(TRAINER_RANKS)
        training_examples = get_training_examples()
        for _ in range(3):
            futures = []
            num_trainers = len(trainer_rrefs)
            for idx, trainer_rref in enumerate(trainer_rrefs):
                trainer_has_less_inputs = simulate_uneven_inputs and idx < num_trainers // 2
                futures.append(_remote_method_async(Trainer.train_batch, trainer_rref, training_examples[idx], trainer_has_less_inputs, simulate_uneven_inputs))
            for future in futures:
                ddp_grads, non_ddp_grads = future.wait()
                if not simulate_uneven_inputs:
                    for grad in ddp_grads:
                        self.assertEqual(grad, torch.zeros_like(grad), msg=f"The grad for any ddp parameter should be zeros, because the training examples' grads cancel each other. Received gradient {grad}")
                for grad in non_ddp_grads:
                    self.assertNotEqual(grad, torch.zeros_like(grad), msg="The grad for any non-ddp parameter shouldn't be zeros")
        for idx, trainer_rref in enumerate(trainer_rrefs):
            _remote_method_async(Trainer.destroy_pg, trainer_rref).wait()
        for rank in TRAINER_RANKS:
            trainer = self.trainer_name(rank)
            rpc.rpc_sync(trainer, set_shutdown_signal, args=())
        rpc.rpc_sync(self.remote_worker_name(), set_shutdown_signal, args=())

    def _do_test(self, ddp_mode, simulate_uneven_inputs=False):
        if self.rank == MASTER_RANK:
            self._master_process(ddp_mode, simulate_uneven_inputs)
        elif self.rank == REMOTE_WORKER_RANK:
            self._remote_worker_process(ddp_mode)
        elif self.rank in TRAINER_RANKS:
            self._trainer_process(self.rank)
        else:
            raise RuntimeError(f'Unknown process rank: {self.rank}')

    @requires_gloo()
    @dist_init
    def test_backward_no_ddp(self):
        self._do_test(DdpMode.NONE)

    @requires_gloo()
    @dist_init
    def test_backward_ddp_outside(self):
        self._do_test(DdpMode.OUTSIDE)

    @requires_gloo()
    @dist_init
    def test_backward_ddp_outside_uneven_inputs(self):
        self._do_test(DdpMode.OUTSIDE, simulate_uneven_inputs=True)

    @requires_gloo()
    @dist_init
    def test_backward_ddp_inside(self):
        self._do_test(DdpMode.INSIDE)