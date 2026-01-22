from typing import cast
from ..compat import cupy, tensorflow, torch
from ..types import ArrayXd
from ..util import get_torch_default_device, tensorflow2xp
Function that can be passed into cupy.cuda.set_allocator, to have cupy
    allocate memory via PyTorch. This is important when using the two libraries
    together, as otherwise OOM errors can occur when there's available memory
    sitting in the other library's pool.
    