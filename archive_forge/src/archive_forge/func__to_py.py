import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _to_py(self, x):
    if isinstance(x, str):
        return "b'%s'" % (x,)
    if isinstance(x, (list, tuple)):
        rep = [self._to_py(item) for item in x]
        if len(rep) == 1:
            rep.append('')
        return '(%s)' % (','.join(rep),)
    return x.as_python_expr()