import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def ComponentTangle(component_idx):
    """The unknotted (1,1) tangle with a specified component index.
    The component index can be a negative number following the usual
    Python list indexing rules, so -1 means the component containing
    this tangle should be the last component when it is turned into
    a Link.

    >>> T=(RationalTangle(2,3)+IdentityBraid(1))|(RationalTangle(2,5)+ComponentTangle(-1))
    >>> T.describe()
    'Tangle[{1,2}, {3,4}, X[5,6,7,3], X[8,7,6,9], X[1,8,9,5], X[10,11,12,4], X[13,14,11,10], X[15,16,14,17], X[2,15,17,13], P[16,12, component->-1]]'

    >>> M=T.braid_closure().exterior() # doctest: +SNAPPY
    >>> M.dehn_fill([(1,0),(0,0)]) # doctest: +SNAPPY
    >>> M.filled_triangulation().identify() # doctest: +SNAPPY
    [m004(0,0), 4_1(0,0), K2_1(0,0), K4a1(0,0), otet02_00001(0,0)]

    >>> T=(RationalTangle(2,3)+IdentityBraid(1))|(RationalTangle(2,5)+ComponentTangle(0))
    >>> T.describe()
    'Tangle[{1,2}, {3,4}, X[5,6,7,3], X[8,7,6,9], X[1,8,9,5], X[10,11,12,4], X[13,14,11,10], X[15,16,14,17], X[2,15,17,13], P[16,12, component->0]]'

    >>> M=T.braid_closure().exterior() # doctest: +SNAPPY
    >>> M.dehn_fill([(0,0),(1,0)]) # doctest: +SNAPPY
    >>> M.filled_triangulation().identify() # doctest: +SNAPPY
    [m004(0,0), 4_1(0,0), K2_1(0,0), K4a1(0,0), otet02_00001(0,0)]

    >>> T=(RationalTangle(2,3)+ComponentTangle(0))|(RationalTangle(2,5)+ComponentTangle(0))
    >>> T.braid_closure()
    Traceback (most recent call last):
        ...
    ValueError: Two Strand objects in different components have the same component_idx values

    """
    s = Strand(component_idx=component_idx)
    return Tangle((1, 1), [s], [(s, 0), (s, 1)])