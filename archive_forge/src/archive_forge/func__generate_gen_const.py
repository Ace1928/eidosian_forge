import sys, os
import types
from . import model
from .error import VerificationError
def _generate_gen_const(self, is_int, name, tp=None, category='const', check_value=None):
    prnt = self._prnt
    funcname = '_cffi_%s_%s' % (category, name)
    self.export_symbols.append(funcname)
    if check_value is not None:
        assert is_int
        assert category == 'const'
        prnt('int %s(char *out_error)' % funcname)
        prnt('{')
        self._check_int_constant_value(name, check_value)
        prnt('  return 0;')
        prnt('}')
    elif is_int:
        assert category == 'const'
        prnt('int %s(long long *out_value)' % funcname)
        prnt('{')
        prnt('  *out_value = (long long)(%s);' % (name,))
        prnt('  return (%s) <= 0;' % (name,))
        prnt('}')
    else:
        assert tp is not None
        assert check_value is None
        if category == 'var':
            ampersand = '&'
        else:
            ampersand = ''
        extra = ''
        if category == 'const' and isinstance(tp, model.StructOrUnion):
            extra = 'const *'
            ampersand = '&'
        prnt(tp.get_c_name(' %s%s(void)' % (extra, funcname), name))
        prnt('{')
        prnt('  return (%s%s);' % (ampersand, name))
        prnt('}')
    prnt()