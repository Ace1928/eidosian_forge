from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class DodecahedralNonorientableClosedCensus(PlatonicManifoldTable):
    """
        Iterator for the dodecahedral non-orientable closed hyperbolic manifolds up
        to 2 dodecahedra, i.e., manifolds that admit a tessellation by regular finite
        hyperbolic dodecahedra with a dihedral angle of 72 degrees.

        >>> DodecahedralNonorientableClosedCensus[0].volume() # doctest: +NUMERIC6
        22.39812948

        """
    _regex = re.compile('ndodecld\\d+_\\d+')

    def __init__(self, **kwargs):
        return PlatonicManifoldTable.__init__(self, 'dodecahedral_nonorientable_closed_census', **kwargs)