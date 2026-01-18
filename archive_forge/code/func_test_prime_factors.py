from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_prime_factors():
    assert prime_factors(100) == [2, 2, 5, 5]
    assert prime_factors(1) == []