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
def _count_arrays_inner(blocks, typemap):
    ret_count = 0
    arr_set = set()
    for label, block in blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, numba.parfors.parfor.Parfor):
                parfor_blocks = inst.loop_body.copy()
                parfor_blocks[0] = inst.init_block
                ret_count += _count_arrays_inner(parfor_blocks, typemap)
            if isinstance(inst, ir.Assign) and isinstance(typemap[inst.target.name], types.ArrayCompatible):
                arr_set.add(inst.target.name)
    ret_count += len(arr_set)
    return ret_count