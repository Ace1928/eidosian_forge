from argparse import ArgumentParser
from collections import namedtuple
from collections.abc import Iterable
import torch
import torch.fft
from torch.utils import benchmark
from torch.utils.benchmark.op_fuzzers.spectral import SpectralOpFuzzer
Microbenchmarks for the torch.fft module