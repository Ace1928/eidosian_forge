import copy
import operator
import types as pytypes
import operator
import warnings
from dataclasses import make_dataclass
import llvmlite.ir
import numpy as np
import numba
from numba.parfors import parfor
from numba.core import types, ir, config, compiler, sigutils, cgutils
from numba.core.ir_utils import (
from numba.core.typing import signature
from numba.core import lowering
from numba.parfors.parfor import ensure_parallel_support
from numba.core.errors import (
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder
@property
def _disable_sroa_like_opt(self):
    """
        Force disable this because Parfor use-defs is incompatible---it only
        considers use-defs in blocks that must be executing.
        See https://github.com/numba/numba/commit/017e2ff9db87fc34149b49dd5367ecbf0bb45268
        """
    return True