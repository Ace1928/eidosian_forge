from contextlib import contextmanager
from typing import Generator, List, Union, cast
import torch
@contextmanager
def use_device(device: torch.device) -> Generator[None, None, None]:
    """:func:`torch.cuda.device` for either CPU or CUDA device."""
    if device.type != 'cuda':
        yield
        return
    with torch.cuda.device(device):
        yield