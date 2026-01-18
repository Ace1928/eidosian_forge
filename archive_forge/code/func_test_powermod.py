from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_powermod():
    assert powermod(3, 8, 7) == 2
    assert powermod(3, Integer(11) / 2, 13) in [3, 10]