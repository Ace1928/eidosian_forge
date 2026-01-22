import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def MinusOneTangle():
    """The minus one tangle, equivalent to ``RationalTangle(-1)``."""
    c = Crossing('-one')
    return Tangle(2, [c], [(c, 3), (c, 0), (c, 2), (c, 1)], 'MinusOneTangle')