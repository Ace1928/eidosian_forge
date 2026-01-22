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
def allocation_hoist(self):
    found = False
    print('Allocation hoisting:')
    for pf_id, data in self.hoist_info.items():
        stmt = data.get('hoisted', [])
        for inst in stmt:
            if isinstance(inst.value, ir.Expr):
                try:
                    attr = inst.value.attr
                    if attr == 'empty':
                        msg = 'The memory allocation derived from the instruction at %s is hoisted out of the parallel loop labelled #%s (it will be performed before the loop is executed and reused inside the loop):'
                        loc = inst.loc
                        print_wrapped(msg % (loc, pf_id))
                        try:
                            path = os.path.relpath(loc.filename)
                        except ValueError:
                            path = os.path.abspath(loc.filename)
                        lines = linecache.getlines(path)
                        if lines and loc.line:
                            print_wrapped('   Allocation:: ' + lines[0 if loc.line < 2 else loc.line - 1].strip())
                        print_wrapped('    - numpy.empty() is used for the allocation.\n')
                        found = True
                except (KeyError, AttributeError):
                    pass
    if not found:
        print_wrapped('No allocation hoisting found')