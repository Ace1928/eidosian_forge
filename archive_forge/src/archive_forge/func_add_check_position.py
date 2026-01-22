import types as pytypes  # avoid confusion with numba.types
import sys, math
import os
import textwrap
import copy
import inspect
import linecache
from functools import reduce
from collections import defaultdict, OrderedDict, namedtuple
from contextlib import contextmanager
import operator
from dataclasses import make_dataclass
import warnings
from llvmlite import ir as lir
from numba.core.imputils import impl_ret_untracked
import numba.core.ir
from numba.core import types, typing, utils, errors, ir, analysis, postproc, rewrites, typeinfer, config, ir_utils
from numba import prange, pndindex
from numba.np.npdatetime_helpers import datetime_minimum, datetime_maximum
from numba.np.numpy_support import as_dtype, numpy_version
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.stencils.stencilparfor import StencilPass
from numba.core.extending import register_jitable, lower_builtin
from numba.core.ir_utils import (
from numba.core.analysis import (compute_use_defs, compute_live_map,
from numba.core.controlflow import CFGraph
from numba.core.typing import npydecl, signature
from numba.core.types.functions import Function
from numba.parfors.array_analysis import (random_int_args, random_1arg_size,
from numba.core.extending import overload
import copy
import numpy
import numpy as np
from numba.parfors import array_analysis
import numba.cpython.builtins
from numba.stencils import stencilparfor
def add_check_position(new_position, array_accessed, index_positions, indexed_arrays, non_indexed_arrays):
    """Returns True if there is a reason to prevent fusion based
           on the rules described above.
           new_position will be a list or tuple of booleans that
           says whether the index in that spot is a parfor index
           or not.  array_accessed is the array on which the access
           is occurring."""
    if isinstance(new_position, list):
        new_position = tuple(new_position)
    if True not in new_position:
        if array_accessed in indexed_arrays:
            return True
        else:
            non_indexed_arrays.add(array_accessed)
            return False
    if array_accessed in non_indexed_arrays:
        return True
    indexed_arrays.add(array_accessed)
    npsize = len(new_position)
    if npsize not in index_positions:
        index_positions[npsize] = new_position
        return False
    return index_positions[npsize] != new_position