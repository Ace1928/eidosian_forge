from math import floor
from ....base import numeric_types
from ...rnn import HybridRecurrentCell
def _get_conv_out_size(dimensions, kernels, paddings, dilations):
    return tuple((int(floor(x + 2 * p - d * (k - 1) - 1) + 1) if x else 0 for x, k, p, d in zip(dimensions, kernels, paddings, dilations)))