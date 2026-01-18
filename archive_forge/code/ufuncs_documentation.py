import math
import numpy as np
from functools import lru_cache
from numba.core import typing
from numba.cuda.mathimpl import (get_unary_impl_for_fn_and_ty,
Contains information on how to translate different ufuncs for the CUDA
target. It is a database of different ufuncs and how each of its loops maps to
a function that implements the inner kernel of that ufunc (the inner kernel
being the per-element function).

Use get_ufunc_info() to get the information related to a ufunc.
