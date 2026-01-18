from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def transfer_arcs_from_R3(self, arcs):
    return [arc.transform_to_R4(self.inverse_matrix) for arc in arcs]