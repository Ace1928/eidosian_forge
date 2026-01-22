import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _struct_names(self, tp):
    cname = tp.get_c_name('')
    if ' ' in cname:
        return (cname, cname.replace(' ', '_'))
    else:
        return (cname, '_' + cname)