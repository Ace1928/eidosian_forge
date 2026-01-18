from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_rsa_exhaustive():
    p, q = (61, 53)
    e = 17
    puk = rsa_public_key(p, q, e, totient='Carmichael')
    prk = rsa_private_key(p, q, e, totient='Carmichael')
    for msg in range(puk[0]):
        encrypted = encipher_rsa(msg, puk)
        decrypted = decipher_rsa(encrypted, prk)
        try:
            assert decrypted == msg
        except AssertionError:
            raise AssertionError('The RSA is not correctly decrypted (Original : {}, Encrypted : {}, Decrypted : {})'.format(msg, encrypted, decrypted))