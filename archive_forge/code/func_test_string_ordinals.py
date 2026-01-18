from sympy.sets.ordinals import Ordinal, OmegaPower, ord0, omega
from sympy.testing.pytest import raises
def test_string_ordinals():
    assert str(omega) == 'w'
    assert str(Ordinal(OmegaPower(5, 3), OmegaPower(3, 2))) == 'w**5*3 + w**3*2'
    assert str(Ordinal(OmegaPower(5, 3), OmegaPower(0, 5))) == 'w**5*3 + 5'
    assert str(Ordinal(OmegaPower(1, 3), OmegaPower(0, 5))) == 'w*3 + 5'
    assert str(Ordinal(OmegaPower(omega + 1, 1), OmegaPower(3, 2))) == 'w**(w + 1) + w**3*2'