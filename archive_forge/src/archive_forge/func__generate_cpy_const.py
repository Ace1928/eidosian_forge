import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_const(self, is_int, name, tp=None, category='const', check_value=None):
    if (category, name) in self._seen_constants:
        raise VerificationError("duplicate declaration of %s '%s'" % (category, name))
    self._seen_constants.add((category, name))
    prnt = self._prnt
    funcname = '_cffi_%s_%s' % (category, name)
    if is_int:
        prnt('static int %s(unsigned long long *o)' % funcname)
        prnt('{')
        prnt('  int n = (%s) <= 0;' % (name,))
        prnt('  *o = (unsigned long long)((%s) | 0);  /* check that %s is an integer */' % (name, name))
        if check_value is not None:
            if check_value > 0:
                check_value = '%dU' % (check_value,)
            prnt('  if (!_cffi_check_int(*o, n, %s))' % (check_value,))
            prnt('    n |= 2;')
        prnt('  return n;')
        prnt('}')
    else:
        assert check_value is None
        prnt('static void %s(char *o)' % funcname)
        prnt('{')
        prnt('  *(%s)o = %s;' % (tp.get_c_name('*'), name))
        prnt('}')
    prnt()