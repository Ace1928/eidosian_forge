from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class OctahedralNonorientableCuspedCensus(PlatonicManifoldTable):
    """
        Iterator for the octahedral non-orientable cusped hyperbolic manifolds up to
        5 octahedra, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic octahedra.

        >>> for M in OctahedralNonorientableCuspedCensus(solids = 3, betti = 3,cusps = 4):
        ...     print(M, M.homology())
        noct03_00007(0,0)(0,0)(0,0)(0,0) Z/2 + Z + Z + Z
        noct03_00029(0,0)(0,0)(0,0)(0,0) Z/2 + Z + Z + Z
        noct03_00047(0,0)(0,0)(0,0)(0,0) Z/2 + Z + Z + Z
        noct03_00048(0,0)(0,0)(0,0)(0,0) Z/2 + Z + Z + Z

        """
    _regex = re.compile('noct\\d+_\\d+')

    def __init__(self, **kwargs):
        return PlatonicManifoldTable.__init__(self, 'octahedral_nonorientable_cusped_census', **kwargs)