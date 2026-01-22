import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
import re
from collections import namedtuple
from itertools import chain
from typing import (
import sympy
from sympy.printing.printer import Printer
import torch
import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from .. import config, metrics
from ..utils import (
from ..virtualized import ops, OpsValue, V
class CppWrapperKernelArgs(KernelArgs):

    def wrap_ptr_arg(self, buf, dtype):
        from .cpp import DTYPE_TO_CPP
        if config.aot_inductor.abi_compatible:
            return buf
        else:
            return f'({DTYPE_TO_CPP[dtype]}*)({buf}.data_ptr())'

    def wrap_size_arg(self, size):
        return f'{size}'