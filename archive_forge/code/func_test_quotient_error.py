from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_quotient_error():
    raises(ZeroDivisionError, lambda: quotient(1, 0))