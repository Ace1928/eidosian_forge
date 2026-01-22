from argparse import ArgumentParser
from collections import namedtuple
from collections.abc import Iterable
import torch
import torch.fft
from torch.utils import benchmark
from torch.utils.benchmark.op_fuzzers.spectral import SpectralOpFuzzer
def _dim_options(ndim):
    if ndim == 1:
        return [None]
    elif ndim == 2:
        return [0, 1, None]
    elif ndim == 3:
        return [0, 1, 2, (0, 1), (0, 2), None]
    raise ValueError(f'Expected ndim in range 1-3, got {ndim}')