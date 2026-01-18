import torch
from accelerate import PartialState
from accelerate.test_utils.testing import assert_exception
from accelerate.utils.dataclasses import DistributedType
from accelerate.utils.operations import (
def test_op_checker(state):
    if state.distributed_type in [DistributedType.NO, DistributedType.XLA]:
        return
    state.debug = True
    if state.process_index == 0:
        data = {'tensor': torch.tensor([[0.0, 1, 2, 3, 4]]).to(state.device)}
    else:
        data = {'tensor': torch.tensor([[[0.0, 1, 2, 3, 4, 5]]]).to(state.device)}
    with assert_exception(DistributedOperationException):
        pad_across_processes(data, dim=0)
    if state.process_index == 0:
        data = {'tensor': torch.tensor([[0.0, 1, 2, 3, 4]]).to(state.device)}
    else:
        data = {'tensor': torch.tensor([[[0.0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]]).to(state.device)}
    with assert_exception(DistributedOperationException):
        reduce(data)
    if state.process_index == 0:
        data = {'tensor': torch.tensor([[0.0, 1, 2, 3, 4]]).to(state.device)}
    else:
        data = {'tensor': torch.tensor([[[0.0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]]).to(state.device)}
    with assert_exception(DistributedOperationException):
        broadcast(data)
    state.debug = False