from __future__ import annotations
import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import textwrap
from typing import Any, Counter, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch._logging
from torch._prims_common import is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges
from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..codecache import code_hash, get_path, PyCodeCache
from ..dependencies import MemoryDep, StarDep
from ..ir import IRNode, ReductionHint, TritonTemplateBuffer
from ..optimize_indexing import indexing_dtype_strength_reduction
from ..scheduler import BaseScheduling, WhyNoFuse
from ..triton_heuristics import AutotuneHint
from ..utils import (
from ..virtualized import ops, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .common import (
from .triton_utils import config_of, signature_of, signature_to_meta
def triton_compute_type(dtype):
    triton_type_name = str(dtype).split('.')[-1]
    if triton_type_name == 'bool':
        triton_type_name = 'int1'
    elif triton_type_name in ('float16', 'bfloat16'):
        triton_type_name = 'float32'
    elif triton_type_name == 'float8_e4m3fn':
        triton_type_name = 'float8e4nv'
    elif triton_type_name == 'float8_e5m2':
        triton_type_name = 'float8e5'
    return f'tl.{triton_type_name}'