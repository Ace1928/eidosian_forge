from ..block import HybridBlock
from ... import symbol
from ...base import numeric_types
from .activations import Activation
from ...util import is_np_array
class MaxPool1D(_Pooling):
    """Max pooling operation for one dimensional data.


    Parameters
    ----------
    pool_size: int
        Size of the max pooling windows.
    strides: int, or None
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCW'
        Dimension ordering of data and out ('NCW' or 'NWC').
        'N', 'C', 'W' stands for batch, channel, and width (time) dimensions
        respectively. Pooling is applied on the W dimension.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.


    Inputs:
        - **data**: 3D input tensor with shape `(batch_size, in_channels, width)`
          when `layout` is `NCW`. For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 3D output tensor with shape `(batch_size, channels, out_width)`
          when `layout` is `NCW`. out_width is calculated as::

              out_width = floor((width+2*padding-pool_size)/strides)+1

          When `ceil_mode` is `True`, ceil will be used instead of floor in this
          equation.
    """

    def __init__(self, pool_size=2, strides=None, padding=0, layout='NCW', ceil_mode=False, **kwargs):
        assert layout in ('NCW', 'NWC'), 'Only NCW and NWC layouts are valid for 1D Pooling'
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,)
        assert len(pool_size) == 1, 'pool_size must be a number or a list of 1 ints'
        super(MaxPool1D, self).__init__(pool_size, strides, padding, ceil_mode, False, 'max', layout, **kwargs)