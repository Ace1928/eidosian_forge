from .module import Module
from .utils import _pair, _quadruple, _ntuple
from .. import functional as F
from torch import Tensor
from ..common_types import _size_2_t, _size_4_t, _size_6_t
from typing import Sequence, Tuple
class CircularPad2d(_CircularPadNd):
    """Pads the input tensor using circular padding of the input boundary.

    Tensor values at the beginning of the dimension are used to pad the end,
    and values at the end are used to pad the beginning. If negative padding is
    applied then the ends of the tensor get removed.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\\text{padding\\_left}`,
            :math:`\\text{padding\\_right}`, :math:`\\text{padding\\_top}`, :math:`\\text{padding\\_bottom}`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \\text{padding\\_top} + \\text{padding\\_bottom}`

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> m = nn.CircularPad2d(2)
        >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> m(input)
        tensor([[[[4., 5., 3., 4., 5., 3., 4.],
                  [7., 8., 6., 7., 8., 6., 7.],
                  [1., 2., 0., 1., 2., 0., 1.],
                  [4., 5., 3., 4., 5., 3., 4.],
                  [7., 8., 6., 7., 8., 6., 7.],
                  [1., 2., 0., 1., 2., 0., 1.],
                  [4., 5., 3., 4., 5., 3., 4.]]]])
        >>> # using different paddings for different sides
        >>> m = nn.CircularPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[5., 3., 4., 5., 3.],
                  [8., 6., 7., 8., 6.],
                  [2., 0., 1., 2., 0.],
                  [5., 3., 4., 5., 3.],
                  [8., 6., 7., 8., 6.]]]])
    """
    padding: Tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        super().__init__()
        self.padding = _quadruple(padding)

    def _check_input_dim(self, input):
        if input.dim() != 3 and input.dim() != 4:
            raise ValueError(f'expected 3D or 4D input (got {input.dim()}D input)')