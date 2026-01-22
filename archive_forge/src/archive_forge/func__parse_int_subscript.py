import copy
import itertools
import operator
import string
import warnings
import cupy
from cupy._core import _accelerator
from cupy import _util
from cupy.linalg._einsum_opt import _greedy_path
from cupy.linalg._einsum_opt import _optimal_path
from cupy.linalg._einsum_cutn import _try_use_cutensornet
def _parse_int_subscript(list_subscript):
    str_subscript = ''
    for s in list_subscript:
        if s is Ellipsis:
            str_subscript += '@'
        else:
            try:
                s = operator.index(s)
            except TypeError as e:
                raise TypeError('For this input type lists must contain either int or Ellipsis') from e
            str_subscript += einsum_symbols[s]
    return str_subscript