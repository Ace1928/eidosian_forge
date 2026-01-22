from typing import List, Optional, Tuple
import torch
from torch import Tensor
from ..copy import Context as CopyContext
from ..copy import Copy
from ..phony import get_phony
from ..stream import AbstractStream, get_device
class PortalBlue(torch.autograd.Function):
    """Hides a tensor from the autograd engine by a :class:`Portal`."""

    @staticmethod
    def forward(ctx: Context, portal: Portal, tensor: Tensor) -> Tensor:
        ctx.portal = portal
        phony = get_phony(tensor.device, requires_grad=False)
        return phony.detach()

    @staticmethod
    def backward(ctx: Context, grad_phony: Tensor) -> Tuple[None, Tensor]:
        grad = ctx.portal.use_grad()
        return (None, grad)