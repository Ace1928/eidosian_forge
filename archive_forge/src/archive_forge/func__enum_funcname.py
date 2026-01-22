import sys, os
import types
from . import model
from .error import VerificationError
def _enum_funcname(self, prefix, name):
    name = name.replace('$', '___D_')
    return '_cffi_e_%s_%s' % (prefix, name)