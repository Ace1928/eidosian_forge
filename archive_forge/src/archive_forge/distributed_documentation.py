from typing import Optional
import torch
from torch import Tensor
from torch.distributed import ProcessGroup
Get the dim for the local rank derived from splitting dim on world_size processes.

    The split may not be even across the world_size processes.
    