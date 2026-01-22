import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def ZeroTangle():
    """The zero tangle, equivalent to ``RationalTangle(0)`` or
    ``CupTangle() * CapTangle()``."""
    bot, top = (Strand('B'), Strand('T'))
    return Tangle(2, [bot, top], [(bot, 0), (bot, 1), (top, 0), (top, 1)], 'ZeroTangle')