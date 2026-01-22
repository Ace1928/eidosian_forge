import sys, os
import types
from . import model
from .error import VerificationError
def _generate_gen_enum_decl(self, tp, name, prefix='enum'):
    if tp.partial:
        for enumerator in tp.enumerators:
            self._generate_gen_const(True, enumerator)
        return
    funcname = self._enum_funcname(prefix, name)
    self.export_symbols.append(funcname)
    prnt = self._prnt
    prnt('int %s(char *out_error)' % funcname)
    prnt('{')
    for enumerator, enumvalue in zip(tp.enumerators, tp.enumvalues):
        self._check_int_constant_value(enumerator, enumvalue)
    prnt('  return 0;')
    prnt('}')
    prnt()