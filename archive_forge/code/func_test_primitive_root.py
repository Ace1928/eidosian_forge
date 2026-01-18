from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_primitive_root():
    assert primitive_root(27) in [2, 5, 11, 14, 20, 23]
    assert primitive_root(15) is None