from ..block import HybridBlock
from ... import symbol
from ...base import numeric_types
from .activations import Activation
from ...util import is_np_array
class GlobalAvgPool2D(_Pooling):
    """Global average pooling operation for spatial data.

    Parameters
    ----------
    layout : str, default 'NCHW'
        Dimension ordering of data and out ('NCHW' or 'NHWC').
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively.


    Inputs:
        - **data**: 4D input tensor with shape
          `(batch_size, in_channels, height, width)` when `layout` is `NCHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 4D output tensor with shape
          `(batch_size, channels, 1, 1)` when `layout` is `NCHW`.
    """

    def __init__(self, layout='NCHW', **kwargs):
        assert layout in ('NCHW', 'NHWC'), 'Only NCHW and NHWC layouts are valid for 2D Pooling'
        super(GlobalAvgPool2D, self).__init__((1, 1), None, 0, True, True, 'avg', layout, **kwargs)