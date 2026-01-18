from .module import Module
from .. import functional as F
from torch import Tensor
from typing import Optional
from ..common_types import _size_2_t, _ratio_2_t, _size_any_t, _ratio_any_t
Applies a 2D bilinear upsampling to an input signal composed of several input channels.

    To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
    as it's constructor argument.

    When :attr:`size` is given, it is the output size of the image `(h, w)`.

    Args:
        size (int or Tuple[int, int], optional): output spatial sizes
        scale_factor (float or Tuple[float, float], optional): multiplier for
            spatial size.

    .. warning::
        This class is deprecated in favor of :func:`~nn.functional.interpolate`. It is
        equivalent to ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where

    .. math::
        H_{out} = \left\lfloor H_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
        W_{out} = \left\lfloor W_{in} \times \text{scale\_factor} \right\rfloor

    Examples::

        >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
        >>> input
        tensor([[[[1., 2.],
                  [3., 4.]]]])

        >>> # xdoctest: +IGNORE_WANT("do other tests modify the global state?")
        >>> m = nn.UpsamplingBilinear2d(scale_factor=2)
        >>> m(input)
        tensor([[[[1.0000, 1.3333, 1.6667, 2.0000],
                  [1.6667, 2.0000, 2.3333, 2.6667],
                  [2.3333, 2.6667, 3.0000, 3.3333],
                  [3.0000, 3.3333, 3.6667, 4.0000]]]])
    