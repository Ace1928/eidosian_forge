from .spatial_dict import SpatialDict, floor_as_integers
from .line import R13Line, R13LineWithMatrix
from .geodesic_info import LiftedTetrahedron
from ..hyperboloid import ( # type: ignore
from ..snap.t3mlite import Mcomplex # type: ignore
from ..matrix import matrix # type: ignore
class ZQuotientLiftedTetrahedronSet:

    def __init__(self, mcomplex: Mcomplex, line_with_matrix: R13LineWithMatrix):
        self._dict = _ZQuotientDict(mcomplex, line_with_matrix)
        self._mcomplex = mcomplex

    def add(self, lifted_tetrahedron: LiftedTetrahedron) -> bool:
        tets = self._dict.setdefault(lifted_tetrahedron.o13_matrix * self._mcomplex.R13_baseTetInCenter, set())
        if lifted_tetrahedron.tet in tets:
            return False
        tets.add(lifted_tetrahedron.tet)
        return True