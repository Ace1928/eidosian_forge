import random
import numpy as np
from ...block import Block, HybridBlock
from ...nn import Sequential, HybridSequential
from .... import image
from ....base import numeric_types
from ....util import is_np_array
class RandomHue(HybridBlock):
    """Randomly jitters image hue with a factor
    chosen from `[max(0, 1 - hue), 1 + hue]`.

    Parameters
    ----------
    hue: float
        How much to jitter hue. hue factor is randomly
        chosen from `[max(0, 1 - hue), 1 + hue]`.


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """

    def __init__(self, hue):
        super(RandomHue, self).__init__()
        self._args = (max(0, 1 - hue), 1 + hue)

    def hybrid_forward(self, F, x):
        if is_np_array():
            F = F.npx
        return F.image.random_hue(x, *self._args)