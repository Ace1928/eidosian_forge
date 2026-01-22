from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class OctahedralOrientableCuspedCensus(PlatonicManifoldTable):
    """
        Iterator for the octahedral orientable cusped hyperbolic manifolds up to
        7 octahedra, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic octahedra.

            >>> OctahedralOrientableCuspedCensus.identify(Manifold("5^2_1"))
            ooct01_00001(0,0)(0,0)

        For octahedral manifolds that are also the complement of an `Augmented
        Knotted Trivalent Graph (AugKTG) <http://arxiv.org/abs/0805.0094>`_, the
        corresponding link is included::

            >>> M = OctahedralOrientableCuspedCensus['ooct04_00034']
            >>> M.link()
            <Link: 4 comp; 17 cross>

        The link can be viewed with ``M.plink()``. To only see complements of
        AugKTGs, supply ``isAugKTG = True``::

         >>> len(OctahedralOrientableCuspedCensus(isAugKTG = True))
         238
         >>> for M in OctahedralOrientableCuspedCensus(isAugKTG = True)[:5]:
         ...     print(M, M.link().DT_code(DT_alpha=True))
         ooct02_00001(0,0)(0,0)(0,0)(0,0) DT[mdbceceJamHBlCKgdfI]
         ooct02_00002(0,0)(0,0)(0,0) DT[lcgbcIkhLBJecGaFD]
         ooct02_00003(0,0)(0,0)(0,0) DT[icebbGIAfhcEdB]
         ooct02_00005(0,0)(0,0)(0,0) DT[hcdbbFHegbDAc]
         ooct04_00027(0,0)(0,0)(0,0)(0,0) DT[zdpecbBujVtiWzsLQpxYREadhOKCmFgN]


        """
    _select = 'select name, triangulation, DT from %s '
    _regex = re.compile('ooct\\d+_\\d+')

    def __init__(self, **kwargs):
        return PlatonicManifoldTable.__init__(self, 'octahedral_orientable_cusped_census', **kwargs)

    def _configure(self, **kwargs):
        PlatonicManifoldTable._configure(self, **kwargs)
        conditions = []
        if 'isAugKTG' in kwargs:
            if kwargs['isAugKTG']:
                conditions.append('isAugKTG = 1')
            else:
                conditions.append('isAugKTG = 0')
        if self._filter:
            if len(conditions) > 0:
                self._filter += ' and ' + ' and '.join(conditions)
        else:
            self._filter = ' and '.join(conditions)

    def _finalize(self, M, row):
        PlatonicManifoldTable._finalize(self, M, row)
        if row[2] and (not row[2] == 'Null'):
            M._set_DTcode(row[2])