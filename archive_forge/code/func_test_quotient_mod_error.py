from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_quotient_mod_error():
    raises(ZeroDivisionError, lambda: quotient_mod(1, 0))