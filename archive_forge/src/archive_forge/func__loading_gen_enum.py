import sys, os
import types
from . import model
from .error import VerificationError
def _loading_gen_enum(self, tp, name, module, prefix='enum'):
    if tp.partial:
        enumvalues = [self._load_constant(True, tp, enumerator, module) for enumerator in tp.enumerators]
        tp.enumvalues = tuple(enumvalues)
        tp.partial_resolved = True
    else:
        funcname = self._enum_funcname(prefix, name)
        self._load_known_int_constant(module, funcname)