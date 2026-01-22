import warnings
from typing import Optional, Tuple
import torch
from torch import Tensor
from .linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from .module import Module
from .. import functional as F
class Hardshrink(Module):
    """Applies the Hard Shrinkage (Hardshrink) function element-wise.

    Hardshrink is defined as:

    .. math::
        \\text{HardShrink}(x) =
        \\begin{cases}
        x, & \\text{ if } x > \\lambda \\\\
        x, & \\text{ if } x < -\\lambda \\\\
        0, & \\text{ otherwise }
        \\end{cases}

    Args:
        lambd: the :math:`\\lambda` value for the Hardshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardshrink.png

    Examples::

        >>> m = nn.Hardshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['lambd']
    lambd: float

    def __init__(self, lambd: float=0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, input: Tensor) -> Tensor:
        return F.hardshrink(input, self.lambd)

    def extra_repr(self) -> str:
        return f'{self.lambd}'