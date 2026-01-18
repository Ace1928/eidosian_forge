import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
@contextlib.contextmanager
def write_to_suffix(self):
    prior = (self.loads, self.compute, self.stores, self.cse)
    self.loads = IndentedBuffer()
    self.compute = IndentedBuffer()
    self.stores = IndentedBuffer()
    self.cse = self.cse.clone()
    yield
    self.reduction_suffix.splice(self.loads)
    self.reduction_suffix.splice(self.compute)
    self.reduction_suffix.splice(self.stores)
    self.loads, self.compute, self.stores, self.cse = prior