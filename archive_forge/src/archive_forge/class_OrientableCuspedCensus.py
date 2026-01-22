from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class OrientableCuspedCensus(ManifoldTable):
    """
        Iterator for all orientable cusped hyperbolic manifolds that
        can be triangulated with at most 9 ideal tetrahedra.

        >>> for M in OrientableCuspedCensus[3:6]: print(M, M.volume()) # doctest: +NUMERIC6
        ... 
        m007(0,0) 2.56897060
        m009(0,0) 2.66674478
        m010(0,0) 2.66674478
        >>> for M in OrientableCuspedCensus[-9:-6]: print(M, M.volume()) # doctest: +NUMERIC6
        ...
        o9_44241(0,0) 8.96323909
        o9_44242(0,0) 8.96736842
        o9_44243(0,0) 8.96736842
        >>> for M in OrientableCuspedCensus[4.10:4.11]: print(M, M.volume()) # doctest: +NUMERIC6
        ... 
        m217(0,0) 4.10795310
        m218(0,0) 4.10942659
        >>> for M in OrientableCuspedCensus(num_cusps=2)[:3]: # doctest: +NUMERIC6
        ...   print(M, M.volume(), M.num_cusps())
        ... 
        m125(0,0)(0,0) 3.66386238 2
        m129(0,0)(0,0) 3.66386238 2
        m202(0,0)(0,0) 4.05976643 2
        >>> M = Manifold('m129')
        >>> M in LinkExteriors
        True
        >>> LinkExteriors.identify(M)
        5^2_1(0,0)(0,0)
        """
    _regex = re.compile('([msvt])([0-9]+)$|o9_\\d\\d\\d\\d\\d$')

    def __init__(self, **kwargs):
        return ManifoldTable.__init__(self, table='orientable_cusped_view', db_path=database_path, **kwargs)