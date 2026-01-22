import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent
from .... import base
def _make_fire(squeeze_channels, expand1x1_channels, expand3x3_channels):
    out = nn.HybridSequential(prefix='')
    out.add(_make_fire_conv(squeeze_channels, 1))
    paths = HybridConcurrent(axis=1, prefix='')
    paths.add(_make_fire_conv(expand1x1_channels, 1))
    paths.add(_make_fire_conv(expand3x3_channels, 3, 1))
    out.add(paths)
    return out