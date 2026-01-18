from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def transfer_arcs_to_R3(self, arcs):
    return [arc.transform_to_R3(self.matrix, bdry_map=self.bdry_map) for arc in arcs]