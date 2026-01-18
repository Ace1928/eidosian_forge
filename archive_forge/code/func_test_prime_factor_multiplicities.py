from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_prime_factor_multiplicities():
    assert prime_factor_multiplicities(90) == {Integer(2): 1, Integer(3): 2, Integer(5): 1}
    assert prime_factor_multiplicities(1) == {}