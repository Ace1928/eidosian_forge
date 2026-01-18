from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_nthroot_mod_list():
    assert nthroot_mod_list(-4, 4, 65) == [4, 6, 7, 9, 17, 19, 22, 32, 33, 43, 46, 48, 56, 58, 59, 61]
    assert nthroot_mod_list(2, 3, 7) == []