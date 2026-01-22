import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _patch_meth(patchlist, cls, name, new_meth):
    old = getattr(cls, name)
    patchlist.append((cls, name, old))
    setattr(cls, name, new_meth)
    return old