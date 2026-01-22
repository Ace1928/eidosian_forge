from typing import Iterable
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch.optim import Optimizer
from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_fabric.utilities.types import _DEVICE
def _optimizers_to_device(optimizers: Iterable[Optimizer], device: _DEVICE) -> None:
    """Moves optimizer states for a sequence of optimizers to the device."""
    for opt in optimizers:
        _optimizer_to_device(opt, device)