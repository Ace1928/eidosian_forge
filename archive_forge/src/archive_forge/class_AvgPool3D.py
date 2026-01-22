from ..block import HybridBlock
from ... import symbol
from ...base import numeric_types
from .activations import Activation
from ...util import is_np_array
class AvgPool3D(_Pooling):
    """Average pooling operation for 3D data (spatial or spatio-temporal).

    Parameters
    ----------
    pool_size: int or list/tuple of 3 ints,
        Size of the average pooling windows.
    strides: int, list/tuple of 3 ints, or None.
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int or list/tuple of 3 ints,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCDHW'
        Dimension ordering of data and out ('NCDHW' or 'NDHWC').
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. padding is applied on 'D', 'H' and 'W'
        dimension.
    ceil_mode : bool, default False
        When True, will use ceil instead of floor to compute the output shape.
    count_include_pad : bool, default True
        When 'False', will exclude padding elements when computing the average value.


    Inputs:
        - **data**: 5D input tensor with shape
          `(batch_size, in_channels, depth, height, width)` when `layout` is `NCDHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 5D output tensor with shape
          `(batch_size, channels, out_depth, out_height, out_width)` when `layout` is `NCDHW`.
          out_depth, out_height and out_width are calculated as::

              out_depth = floor((depth+2*padding[0]-pool_size[0])/strides[0])+1
              out_height = floor((height+2*padding[1]-pool_size[1])/strides[1])+1
              out_width = floor((width+2*padding[2]-pool_size[2])/strides[2])+1

          When `ceil_mode` is `True,` ceil will be used instead of floor in this
          equation.
    """

    def __init__(self, pool_size=(2, 2, 2), strides=None, padding=0, ceil_mode=False, layout='NCDHW', count_include_pad=True, **kwargs):
        assert layout in ('NCDHW', 'NDHWC'), 'Only NCDHW and NDHWC layouts are valid for 3D Pooling'
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,) * 3
        assert len(pool_size) == 3, 'pool_size must be a number or a list of 3 ints'
        super(AvgPool3D, self).__init__(pool_size, strides, padding, ceil_mode, False, 'avg', layout, count_include_pad, **kwargs)