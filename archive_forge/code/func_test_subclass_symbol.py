from symengine.sympy_compat import (Integer, Rational, S, Basic, Add, Mul,
from symengine.test_utilities import raises
def test_subclass_symbol():

    class Wrapper(Symbol):

        def __new__(cls, name, extra_attribute):
            return Symbol.__new__(cls, name)

        def __init__(self, name, extra_attribute):
            super().__init__(name)
            self.extra_attribute = extra_attribute
    x = Wrapper('x', extra_attribute=3)
    assert x.extra_attribute == 3
    two_x = 2 * x
    assert two_x.args[1] is x
    del two_x
    x._unsafe_reset()