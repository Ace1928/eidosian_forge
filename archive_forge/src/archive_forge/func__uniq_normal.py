import common_z3 as CM_Z3
import ctypes
from .z3 import *
def _uniq_normal(seq):
    d_ = {}
    for s in seq:
        if s not in d_:
            d_[s] = None
            yield s