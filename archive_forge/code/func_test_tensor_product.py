from sympy.physics.quantum.hilbert import (
from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.printing.repr import srepr
from sympy.printing.str import sstr
from sympy.sets.sets import Interval
def test_tensor_product():
    n = Symbol('n')
    hs1 = ComplexSpace(2)
    hs2 = ComplexSpace(n)
    h = hs1 * hs2
    assert isinstance(h, TensorProductHilbertSpace)
    assert h.dimension == 2 * n
    assert h.spaces == (hs1, hs2)
    h = hs2 * hs2
    assert isinstance(h, TensorPowerHilbertSpace)
    assert h.base == hs2
    assert h.exp == 2
    assert h.dimension == n ** 2
    f = FockSpace()
    h = hs1 * hs2 * f
    assert h.dimension is oo