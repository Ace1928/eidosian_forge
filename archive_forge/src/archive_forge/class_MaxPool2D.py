from ..block import HybridBlock
from ... import symbol
from ...base import numeric_types
from .activations import Activation
from ...util import is_np_array
class MaxPool2D(_Pooling):
    """Max pooling operation for two dimensional (spatial) data.


    Parameters
    ----------
    pool_size: int or list/tuple of 2 ints,
        Size of the max pooling windows.
    strides: int, list/tuple of 2 ints, or None.
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int or list/tuple of 2 ints,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCHW'
        Dimension ordering of data and out ('NCHW' or 'NHWC').
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively. padding is applied on 'H' and 'W' dimension.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.


    Inputs:
        - **data**: 4D input tensor with shape
          `(batch_size, in_channels, height, width)` when `layout` is `NCHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 4D output tensor with shape
          `(batch_size, channels, out_height, out_width)` when `layout` is `NCHW`.
          out_height and out_width are calculated as::

              out_height = floor((height+2*padding[0]-pool_size[0])/strides[0])+1
              out_width = floor((width+2*padding[1]-pool_size[1])/strides[1])+1

          When `ceil_mode` is `True`, ceil will be used instead of floor in this
          equation.
    """

    def __init__(self, pool_size=(2, 2), strides=None, padding=0, layout='NCHW', ceil_mode=False, **kwargs):
        assert layout in ('NCHW', 'NHWC'), 'Only NCHW and NHWC layouts are valid for 2D Pooling'
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,) * 2
        assert len(pool_size) == 2, 'pool_size must be a number or a list of 2 ints'
        super(MaxPool2D, self).__init__(pool_size, strides, padding, ceil_mode, False, 'max', layout, **kwargs)