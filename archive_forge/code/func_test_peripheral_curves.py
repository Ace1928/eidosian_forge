from ... import sage_helper
from .. import t3mlite as t3m
from . import link, dual_cellulation
def test_peripheral_curves(n=100, progress=True):
    """
    TESTS::

        sage: test_peripheral_curves(5, False)
    """
    import snappy
    census = snappy.OrientableCuspedCensus(cusps=1)
    for i in range(n):
        M = census.random()
        if progress:
            print(M.name())
        peripheral_curve_package(M)