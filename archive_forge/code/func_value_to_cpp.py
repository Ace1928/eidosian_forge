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
def value_to_cpp(value, cpp_type):
    if value == float('-inf'):
        return f'-std::numeric_limits<{cpp_type}>::infinity()'
    elif value == float('inf'):
        return f'std::numeric_limits<{cpp_type}>::infinity()'
    elif isinstance(value, bool):
        return f'static_cast<{cpp_type}>({str(value).lower()})'
    elif math.isnan(value):
        return f'std::numeric_limits<{cpp_type}>::quiet_NaN()'
    else:
        return f'static_cast<{cpp_type}>({repr(value)})'