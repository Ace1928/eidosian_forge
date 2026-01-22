from typing import List, Optional
from torch import Tensor
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F
from ..common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
class MaxUnpool2d(_MaxUnpoolNd):
    """Computes a partial inverse of :class:`MaxPool2d`.

    :class:`MaxPool2d` is not fully invertible, since the non-maximal values are lost.

    :class:`MaxUnpool2d` takes in as input the output of :class:`MaxPool2d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    Note:
        This operation may behave nondeterministically when the input indices has repeat values.
        See https://github.com/pytorch/pytorch/issues/80827 and :doc:`/notes/randomness` for more information.

    .. note:: :class:`MaxPool2d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs and Example below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        stride (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool2d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
            H_{out} = (H_{in} - 1) \\times \\text{stride[0]} - 2 \\times \\text{padding[0]} + \\text{kernel\\_size[0]}

          .. math::
            W_{out} = (W_{in} - 1) \\times \\text{stride[1]} - 2 \\times \\text{padding[1]} + \\text{kernel\\_size[1]}

          or as given by :attr:`output_size` in the call operator

    Example::

        >>> pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool2d(2, stride=2)
        >>> input = torch.tensor([[[[ 1.,  2.,  3.,  4.],
                                    [ 5.,  6.,  7.,  8.],
                                    [ 9., 10., 11., 12.],
                                    [13., 14., 15., 16.]]]])
        >>> output, indices = pool(input)
        >>> unpool(output, indices)
        tensor([[[[  0.,   0.,   0.,   0.],
                  [  0.,   6.,   0.,   8.],
                  [  0.,   0.,   0.,   0.],
                  [  0.,  14.,   0.,  16.]]]])
        >>> # Now using output_size to resolve an ambiguous size for the inverse
        >>> input = torch.torch.tensor([[[[ 1.,  2.,  3., 4., 5.],
                                          [ 6.,  7.,  8., 9., 10.],
                                          [11., 12., 13., 14., 15.],
                                          [16., 17., 18., 19., 20.]]]])
        >>> output, indices = pool(input)
        >>> # This call will not work without specifying output_size
        >>> unpool(output, indices, output_size=input.size())
        tensor([[[[ 0.,  0.,  0.,  0.,  0.],
                  [ 0.,  7.,  0.,  9.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.],
                  [ 0., 17.,  0., 19.,  0.]]]])


    """
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t

    def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t]=None, padding: _size_2_t=0) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)

    def forward(self, input: Tensor, indices: Tensor, output_size: Optional[List[int]]=None) -> Tensor:
        return F.max_unpool2d(input, indices, self.kernel_size, self.stride, self.padding, output_size)