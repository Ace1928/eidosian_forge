from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_rsa_public_key():
    assert rsa_public_key(2, 3, 1) == (6, 1)
    assert rsa_public_key(5, 3, 3) == (15, 3)
    with warns(NonInvertibleCipherWarning):
        assert rsa_public_key(2, 2, 1) == (4, 1)
        assert rsa_public_key(8, 8, 8) is False