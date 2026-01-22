import sys
from Cryptodome.Util.py3compat import tobytes, is_native_int
from Cryptodome.Util._raw_api import (backend, load_lib,
from ._IntegerBase import IntegerBase
def _apply_and_return(self, func, term):
    if not isinstance(term, IntegerGMP):
        term = IntegerGMP(term)
    return func(self._mpz_p, term._mpz_p)