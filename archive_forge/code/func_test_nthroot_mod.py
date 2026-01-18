from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_nthroot_mod():
    assert nthroot_mod(12, 5, 77) in [3, 31, 38, 45, 59]
    assert nthroot_mod(3, 2, 5) is None