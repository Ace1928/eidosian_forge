import common_z3 as CM_Z3
import ctypes
from .z3 import *
def _uniq_idfun(seq, idfun):
    d_ = {}
    for s in seq:
        h_ = idfun(s)
        if h_ not in d_:
            d_[h_] = None
            yield s