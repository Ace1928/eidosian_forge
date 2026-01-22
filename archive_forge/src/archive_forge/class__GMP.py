import sys
from Cryptodome.Util.py3compat import tobytes, is_native_int
from Cryptodome.Util._raw_api import (backend, load_lib,
from ._IntegerBase import IntegerBase
class _GMP(object):

    def __getattr__(self, name):
        if name.startswith('mpz_'):
            func_name = '__gmpz_' + name[4:]
        elif name.startswith('gmp_'):
            func_name = '__gmp_' + name[4:]
        else:
            raise AttributeError('Attribute %s is invalid' % name)
        func = getattr(lib, func_name)
        setattr(self, name, func)
        return func