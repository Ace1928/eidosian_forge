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
Given a variable name, if that variable is not a new name
           introduced as the extracted part of a tuple then just return
           the variable loaded from its name.  However, if the variable
           does represent part of a tuple, as recognized by the name of
           the variable being present in the exp_name_to_tuple_var dict,
           then we load the original tuple var instead that we get from
           the dict and then extract the corresponding element of the
           tuple, also stored and returned to use in the dict (i.e., offset).
        