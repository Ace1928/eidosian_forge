import warnings
from typing import Optional, Tuple
import torch
from torch import Tensor
from .linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from .module import Module
from .. import functional as F
class Softshrink(Module):
    """Applies the soft shrinkage function elementwise:

    .. math::
        \\text{SoftShrinkage}(x) =
        \\begin{cases}
        x - \\lambda, & \\text{ if } x > \\lambda \\\\
        x + \\lambda, & \\text{ if } x < -\\lambda \\\\
        0, & \\text{ otherwise }
        \\end{cases}

    Args:
        lambd: the :math:`\\lambda` (must be no less than zero) value for the Softshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Softshrink.png

    Examples::

        >>> m = nn.Softshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['lambd']
    lambd: float

    def __init__(self, lambd: float=0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, input: Tensor) -> Tensor:
        return F.softshrink(input, self.lambd)

    def extra_repr(self) -> str:
        return str(self.lambd)