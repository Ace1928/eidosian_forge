import warnings
from typing import Optional, Tuple
import torch
from torch import Tensor
from .linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from .module import Module
from .. import functional as F
class RReLU(Module):
    """Applies the randomized leaky rectified linear unit function, element-wise,
    as described in the paper:

    `Empirical Evaluation of Rectified Activations in Convolutional Network`_.

    The function is defined as:

    .. math::
        \\text{RReLU}(x) =
        \\begin{cases}
            x & \\text{if } x \\geq 0 \\\\
            ax & \\text{ otherwise }
        \\end{cases}

    where :math:`a` is randomly sampled from uniform distribution
    :math:`\\mathcal{U}(\\text{lower}, \\text{upper})` during training while during
    evaluation :math:`a` is fixed with :math:`a = \\frac{\\text{lower} + \\text{upper}}{2}`.

     See: https://arxiv.org/pdf/1505.00853.pdf

    Args:
        lower: lower bound of the uniform distribution. Default: :math:`\\frac{1}{8}`
        upper: upper bound of the uniform distribution. Default: :math:`\\frac{1}{3}`
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/RReLU.png

    Examples::

        >>> m = nn.RReLU(0.1, 0.3)
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _`Empirical Evaluation of Rectified Activations in Convolutional Network`:
        https://arxiv.org/abs/1505.00853
    """
    __constants__ = ['lower', 'upper', 'inplace']
    lower: float
    upper: float
    inplace: bool

    def __init__(self, lower: float=1.0 / 8, upper: float=1.0 / 3, inplace: bool=False):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.rrelu(input, self.lower, self.upper, self.training, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return f'lower={self.lower}, upper={self.upper}{inplace_str}'