from typing import List, Optional, Tuple
import torch
from torch import Tensor
from ..copy import Context as CopyContext
from ..copy import Copy
from ..phony import get_phony
from ..stream import AbstractStream, get_device
def use_grad(self) -> Tensor:
    """Retrieves and removes the underlying gradient. The gradient is
        always ephemeral.
        """
    if self.grad is None:
        raise RuntimeError('grad in portal has been removed or never set')
    grad = self.grad
    self.grad = None
    return grad