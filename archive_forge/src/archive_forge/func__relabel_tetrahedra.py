from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.mcomplex import Mcomplex, VERBOSE, edge_and_arrow
from ..snap.t3mlite.tetrahedron import Tetrahedron
def _relabel_tetrahedra(self):
    for i, tet in enumerate(self):
        tet.Index = i