import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def CupTangle():
    """The unknotted (0,2) tangle."""
    cup = Strand('cup')
    return Tangle((0, 2), [cup], [(cup, 0), (cup, 1)])