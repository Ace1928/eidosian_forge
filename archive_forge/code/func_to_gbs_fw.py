from typing import Any, Dict, Optional
import torch
import triton
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components import Activation, build_activation
from xformers.triton import FusedDropoutBias
def to_gbs_fw(a, ms, bias):
    total = 2 * a.numel() * a.element_size()
    if bias:
        total += a.shape[-1] * a.element_size()
    return total * 1e-09 / (ms * 0.001)