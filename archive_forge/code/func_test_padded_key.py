from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_padded_key():
    assert padded_key('b', 'ab') == 'ba'
    raises(ValueError, lambda: padded_key('ab', 'ace'))
    raises(ValueError, lambda: padded_key('ab', 'abba'))