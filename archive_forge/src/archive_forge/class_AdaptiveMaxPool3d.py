from typing import List, Optional
from torch import Tensor
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F
from ..common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
class AdaptiveMaxPool3d(_AdaptiveMaxPoolNd):
    """Applies a 3D adaptive max pooling over an input signal composed of several input planes.

    The output is of size :math:`D_{out} \\times H_{out} \\times W_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form :math:`D_{out} \\times H_{out} \\times W_{out}`.
                     Can be a tuple :math:`(D_{out}, H_{out}, W_{out})` or a single
                     :math:`D_{out}` for a cube :math:`D_{out} \\times D_{out} \\times D_{out}`.
                     :math:`D_{out}`, :math:`H_{out}` and :math:`W_{out}` can be either a
                     ``int``, or ``None`` which means the size will be the same as that of the input.

        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool3d. Default: ``False``

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where :math:`(D_{out}, H_{out}, W_{out})=\\text{output\\_size}`.

    Examples:
        >>> # target output size of 5x7x9
        >>> m = nn.AdaptiveMaxPool3d((5, 7, 9))
        >>> input = torch.randn(1, 64, 8, 9, 10)
        >>> output = m(input)
        >>> # target output size of 7x7x7 (cube)
        >>> m = nn.AdaptiveMaxPool3d(7)
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)
        >>> # target output size of 7x9x8
        >>> m = nn.AdaptiveMaxPool3d((7, None, None))
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)

    """
    output_size: _size_3_opt_t

    def forward(self, input: Tensor):
        return F.adaptive_max_pool3d(input, self.output_size, self.return_indices)