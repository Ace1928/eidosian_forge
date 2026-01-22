from snappy.snap import t3mlite as t3m
from snappy.snap.mcomplex_base import *
from snappy.SnapPy import matrix
from .hyperboloid_utilities import *
def add_weights(self, weights):
    for tet in self.mcomplex.Tetrahedra:
        tet.Weights = {F: weights[4 * tet.Index + f] if weights else 0.0 for f, F in enumerate(t3m.TwoSubsimplices)}