import math
import os
import re
import dis
import numbers
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp
import numba.parfors.parfor
from numba import (njit, prange, parallel_chunksize,
from numba.core import (types, errors, ir, rewrites,
from numba.extending import (overload_method, register_model,
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.compiler import (CompilerBase, DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
from numba.core.extending import register_jitable
from numba.core.bytecode import _fix_LOAD_GLOBAL_arg
from numba.core import utils
import cmath
import unittest
def _count_non_parfor_array_accesses_inner(f_ir, blocks, typemap, parfor_indices=None):
    ret_count = 0
    if parfor_indices is None:
        parfor_indices = set()
    for label, block in blocks.items():
        for stmt in block.body:
            if isinstance(stmt, numba.parfors.parfor.Parfor):
                parfor_indices.add(stmt.index_var.name)
                parfor_blocks = stmt.loop_body.copy()
                parfor_blocks[0] = stmt.init_block
                ret_count += _count_non_parfor_array_accesses_inner(f_ir, parfor_blocks, typemap, parfor_indices)
            elif is_getitem(stmt) and isinstance(typemap[stmt.value.value.name], types.ArrayCompatible) and (not _uses_indices(f_ir, index_var_of_get_setitem(stmt), parfor_indices)):
                ret_count += 1
            elif is_setitem(stmt) and isinstance(typemap[stmt.target.name], types.ArrayCompatible) and (not _uses_indices(f_ir, index_var_of_get_setitem(stmt), parfor_indices)):
                ret_count += 1
            elif isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var) and (stmt.value.name in parfor_indices):
                parfor_indices.add(stmt.target.name)
    return ret_count