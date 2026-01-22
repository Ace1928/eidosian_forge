from collections import defaultdict
import copy
import sys
from itertools import permutations, takewhile
from contextlib import contextmanager
from functools import cached_property
from llvmlite import ir as llvmir
from llvmlite.ir import Constant
import llvmlite.binding as ll
from numba.core import types, utils, datamodel, debuginfo, funcdesc, config, cgutils, imputils
from numba.core import event, errors, targetconfig
from numba import _dynfunc, _helperlib
from numba.core.compiler_lock import global_compiler_lock
from numba.core.pythonapi import PythonAPI
from numba.core.imputils import (user_function, user_generator,
from numba.cpython import builtins
def _sort_signatures(self, candidates):
    """
        Sort signatures in ascending level of genericity.

        Returns a 2-tuple:

            * ordered list of signatures
            * dictionary containing genericity scores
        """
    genericity = defaultdict(int)
    for this, other in permutations(candidates.keys(), r=2):
        matched = self._match_arglist(formal_args=this, actual_args=other)
        if matched:
            genericity[this] += 1
    ordered = sorted(candidates.keys(), key=lambda x: genericity[x])
    return (ordered, genericity)