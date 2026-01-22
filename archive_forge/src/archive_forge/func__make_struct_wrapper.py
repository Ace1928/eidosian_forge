import sys, os
import types
from . import model
from .error import VerificationError
def _make_struct_wrapper(self, oldfunc, i, tp, base_tp):
    backend = self.ffi._backend
    BType = self.ffi._get_cached_btype(tp)
    if i == 'result':
        ffi = self.ffi

        def newfunc(*args):
            res = ffi.new(BType)
            oldfunc(res, *args)
            return res[0]
    else:

        def newfunc(*args):
            args = args[:i] + (backend.newp(BType, args[i]),) + args[i + 1:]
            return oldfunc(*args)
    newfunc._cffi_base_type = base_tp
    return newfunc