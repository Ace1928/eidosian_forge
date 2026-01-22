import contextlib
from typing import Dict, Iterator, Set, Union
import torch
from torch.cuda import _lazy_call
from torch.utils.checkpoint import detach_variable
from .initialize import get_data_parallel_rank, get_model_parallel_rank
def _set_cuda_rng_state(new_state: torch.ByteTensor, device: Union[int, str, torch.device]=-1) -> None:
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if device == -1:
        device = torch.device('cuda')
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('cuda', device)

    def cb() -> None:
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        default_generator = torch.cuda.default_generators[idx]
        default_generator.set_state(new_state)
    _lazy_call(cb)