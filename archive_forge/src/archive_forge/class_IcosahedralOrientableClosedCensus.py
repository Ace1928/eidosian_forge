from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class IcosahedralOrientableClosedCensus(PlatonicManifoldTable):
    """
        Iterator for the icosahedral orientable closed hyperbolic manifolds up
        to 4 icosahedra, i.e., manifolds that admit a tessellation by regula finite
        hyperbolic icosahedra.

        >>> M = IcosahedralOrientableClosedCensus[0]
        >>> M.volume() # doctest: +NUMERIC6
        4.68603427
        >>> M
        oicocld01_00000(1,0)
        """
    _regex = re.compile('oicocld\\d+_\\d+')

    def __init__(self, **kwargs):
        return PlatonicManifoldTable.__init__(self, 'icosahedral_orientable_closed_census', **kwargs)