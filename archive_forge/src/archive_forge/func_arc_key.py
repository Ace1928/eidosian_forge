import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def arc_key(c, i):
    """For the given entity c and index into c.adjacent,
            create a name for the incident arc. This gives something
            that's suitable for use as a dictionary key."""
    d, j = c.adjacent[i]
    return tuple(sorted([(c.label, i), (d.label, j)]))