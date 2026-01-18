from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_rsa_multipower_exhanstive():
    from sympy.core.numbers import igcd
    primes = [5, 5, 7]
    e = 7
    args = primes + [e]
    puk = rsa_public_key(*args, multipower=True)
    prk = rsa_private_key(*args, multipower=True)
    n = puk[0]
    for msg in range(n):
        if igcd(msg, n) != 1:
            continue
        encrypted = encipher_rsa(msg, puk)
        decrypted = decipher_rsa(encrypted, prk)
        try:
            assert decrypted == msg
        except AssertionError:
            raise AssertionError('The RSA is not correctly decrypted (Original : {}, Encrypted : {}, Decrypted : {})'.format(msg, encrypted, decrypted))