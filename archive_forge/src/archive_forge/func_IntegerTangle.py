import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def IntegerTangle(n):
    """The tangle equivalent to ``RationalTangle(n)``. It is
    ``n`` copies of the ``OneTangle`` joined by ``+`` when ``n`` is
    positive, and otherwise ``-n`` copies of ``MinusOneTangle``."""
    if n == 0:
        return ZeroTangle()
    elif n > 0:
        T = OneTangle()
        for i in range(n - 1):
            T += OneTangle()
        T.label = f'IntegerTangle({n})'
        return T
    elif n < 0:
        T = -IntegerTangle(-n)
        T.label = f'IntegerTangle({n})'
        return T
    else:
        raise ValueError('Expecting int')