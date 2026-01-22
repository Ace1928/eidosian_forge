import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _extra_local_variables(self, tp, localvars, freelines):
    if isinstance(tp, model.PointerType):
        localvars.add('Py_ssize_t datasize')
        localvars.add('struct _cffi_freeme_s *large_args_free = NULL')
        freelines.add('if (large_args_free != NULL) _cffi_free_array_arguments(large_args_free);')