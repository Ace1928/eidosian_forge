import warnings
from .... import nd, context
from ...block import HybridBlock, Block
from ...nn import Sequential, HybridSequential, BatchNorm
class PixelShuffle3D(HybridBlock):
    """Pixel-shuffle layer for upsampling in 3 dimensions.

    Pixel-shuffling (or voxel-shuffling in 3D) is the operation of taking
    groups of values along the *channel* dimension and regrouping them into
    blocks of voxels along the ``D``, ``H`` and ``W`` dimensions, thereby
    effectively multiplying those dimensions by a constant factor in size.

    For example, a feature map of shape :math:`(f^3 C, D, H, W)` is reshaped
    into :math:`(C, fD, fH, fW)` by forming little :math:`f \\times f \\times f`
    blocks of voxels and arranging them in a :math:`D \\times H \\times W` grid.

    Pixel-shuffling together with regular convolution is an alternative,
    learnable way of upsampling an image by arbitrary factors. It is reported
    to help overcome checkerboard artifacts that are common in upsampling with
    transposed convolutions (also called deconvolutions). See the paper
    `Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_
    for further details.

    Parameters
    ----------
    factor : int or 3-tuple of int
        Upsampling factors, applied to the ``D``, ``H`` and ``W``
        dimensions, in that order.

    Inputs:
        - **data**: Tensor of shape ``(N, f1*f2*f3*C, D, H, W)``.
    Outputs:
        - **out**: Tensor of shape ``(N, C, D*f1, H*f2, W*f3)``.

    Examples
    --------
    >>> pxshuf = PixelShuffle3D((2, 3, 4))
    >>> x = mx.nd.zeros((1, 48, 3, 5, 7))
    >>> pxshuf(x).shape
    (1, 2, 6, 15, 28)
    """

    def __init__(self, factor):
        super(PixelShuffle3D, self).__init__()
        try:
            self._factors = (int(factor),) * 3
        except TypeError:
            self._factors = tuple((int(fac) for fac in factor))
            assert len(self._factors) == 3, 'wrong length {}'.format(len(self._factors))

    def hybrid_forward(self, F, x):
        """Perform pixel-shuffling on the input."""
        f1, f2, f3 = self._factors
        x = F.reshape(x, (0, -4, -1, f1 * f2 * f3, 0, 0, 0))
        x = F.swapaxes(x, 2, 3)
        x = F.reshape(x, (0, 0, 0, -4, f1, f2 * f3, 0, 0))
        x = F.reshape(x, (0, 0, -3, 0, 0, 0))
        x = F.swapaxes(x, 3, 4)
        x = F.reshape(x, (0, 0, 0, 0, -4, f2, f3, 0))
        x = F.reshape(x, (0, 0, 0, -3, 0, 0))
        x = F.swapaxes(x, 4, 5)
        x = F.reshape(x, (0, 0, 0, 0, -3))
        return x

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._factors)