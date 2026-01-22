from typing import (
from torch import Tensor, nn
from ..microbatch import Batch
from .namespace import Namespace
from .tracker import current_skip_tracker
class stash:
    """The command to stash a skip tensor.

    ::

        def forward(self, input):
            yield stash('name', input)
            return f(input)

    Args:
        name (str): name of skip tensor
        input (torch.Tensor or None): tensor to pass to the skip connection

    """
    __slots__ = ('name', 'tensor')

    def __init__(self, name: str, tensor: Optional[Tensor]) -> None:
        self.name = name
        self.tensor = tensor