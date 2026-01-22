from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class DodecahedralOrientableClosedCensus(PlatonicManifoldTable):
    """
        Iterator for the dodecahedral orientable closed hyperbolic manifolds up
        to 3 dodecahedra, i.e., manifolds that admit a tessellation by regular finite
        hyperbolic dodecahedra with a dihedral angle of 72 degrees.

        The Seifert-Weber space::

          >>> M = DodecahedralOrientableClosedCensus(solids = 1)[-1]
          >>> M.homology()
          Z/5 + Z/5 + Z/5

        """
    _regex = re.compile('ododecld\\d+_\\d+')

    def __init__(self, **kwargs):
        return PlatonicManifoldTable.__init__(self, 'dodecahedral_orientable_closed_census', **kwargs)