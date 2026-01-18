from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_powermod_list():
    assert powermod_list(15, Integer(1) / 6, 21) == [3, 6, 9, 12, 15, 18]
    assert powermod_list(2, Integer(5) / 2, 11) == []