from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_rsa_large_key():
    p = int('101565610013301240713207239558950144682174355406589305284428666903702505233009')
    q = int('89468719188754548893545560595594841381237600305314352142924213312069293984003')
    e = int('65537')
    d = int('8936505818327042395303988587447591295947962354408444794561435666999402846577625762582824202269399672579058991442587406384754958587400493169361356902030209')
    assert rsa_public_key(p, q, e) == (p * q, e)
    assert rsa_private_key(p, q, e) == (p * q, d)