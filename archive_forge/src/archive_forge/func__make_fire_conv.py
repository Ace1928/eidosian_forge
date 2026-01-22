import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent
from .... import base
def _make_fire_conv(channels, kernel_size, padding=0):
    out = nn.HybridSequential(prefix='')
    out.add(nn.Conv2D(channels, kernel_size, padding=padding))
    out.add(nn.Activation('relu'))
    return out