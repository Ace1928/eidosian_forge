import random
import numpy as np
from ...block import Block, HybridBlock
from ...nn import Sequential, HybridSequential
from .... import image
from ....base import numeric_types
from ....util import is_np_array
class RandomSaturation(HybridBlock):
    """Randomly jitters image saturation with a factor
    chosen from `[max(0, 1 - saturation), 1 + saturation]`.

    Parameters
    ----------
    saturation: float
        How much to jitter saturation. saturation factor is randomly
        chosen from `[max(0, 1 - saturation), 1 + saturation]`.


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """

    def __init__(self, saturation):
        super(RandomSaturation, self).__init__()
        self._args = (max(0, 1 - saturation), 1 + saturation)

    def hybrid_forward(self, F, x):
        if is_np_array():
            F = F.npx
        return F.image.random_saturation(x, *self._args)