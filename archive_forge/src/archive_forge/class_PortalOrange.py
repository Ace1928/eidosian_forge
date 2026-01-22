from typing import List, Optional, Tuple
import torch
from torch import Tensor
from ..copy import Context as CopyContext
from ..copy import Copy
from ..phony import get_phony
from ..stream import AbstractStream, get_device
class PortalOrange(torch.autograd.Function):
    """Retrieves the hidden tensor from a :class:`Portal`."""

    @staticmethod
    def forward(ctx: Context, portal: Portal, phony: Tensor) -> Tensor:
        ctx.portal = portal
        tensor = portal.use_tensor()
        assert tensor is not None
        return tensor.detach()

    @staticmethod
    def backward(ctx: Context, grad: Tensor) -> Tuple[None, None]:
        ctx.portal.put_grad(grad)
        return (None, None)