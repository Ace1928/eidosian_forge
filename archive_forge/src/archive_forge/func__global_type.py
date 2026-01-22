import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _global_type(self, tp, global_name):
    if isinstance(tp, model.ArrayType):
        actual_length = tp.length
        if actual_length == '...':
            actual_length = '_cffi_array_len(%s)' % (global_name,)
        tp_item = self._global_type(tp.item, '%s[0]' % global_name)
        tp = model.ArrayType(tp_item, actual_length)
    return tp