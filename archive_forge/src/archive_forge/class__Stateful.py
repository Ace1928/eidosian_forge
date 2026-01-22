from pathlib import Path
from typing import (
import torch
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import TypeAlias, overload
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
@runtime_checkable
class _Stateful(Protocol[_DictKey]):
    """This class is used to detect if an object is stateful using `isinstance(obj, _Stateful)`."""

    def state_dict(self) -> Dict[_DictKey, Any]:
        ...

    def load_state_dict(self, state_dict: Dict[_DictKey, Any]) -> None:
        ...