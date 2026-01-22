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
def argmax_argmin_prefix(reduction_type, src_dtype, tmpvar):
    global index_value_name_counter
    struct_name = f'IndexValue_{index_value_name_counter}'
    index_value_name_counter += 1
    prefix = [f'struct {struct_name} {{size_t index; {DTYPE_TO_CPP[src_dtype]} value;}};', f'{struct_name} {tmpvar}{{0, {reduction_init(reduction_type, src_dtype)}}};']
    if reduction_type == 'argmax':
        prefix.extend(['#if !defined(__clang_major__) || __clang_major__ > 9', f'#pragma omp declare reduction(argmax : {struct_name} :\\', '    omp_out.value = omp_in.value < omp_out.value ? omp_out.value : omp_in.value,\\', '    omp_out.index = omp_in.value < omp_out.value ? omp_out.index : omp_in.index)\\', f'\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})', '#endif'])
    elif reduction_type == 'argmin':
        prefix.extend(['#if !defined(__clang_major__) || __clang_major__ > 9', f'#pragma omp declare reduction(argmin : {struct_name} :\\', '    omp_out.value = omp_in.value > omp_out.value ? omp_out.value : omp_in.value,\\', '    omp_out.index = omp_in.value > omp_out.value ? omp_out.index : omp_in.index)\\', f'\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})', '#endif'])
    return prefix