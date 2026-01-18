from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def transform_to_R4(self, matrix):
    new_start = self.start.transform_to_R4(matrix)
    new_end = self.end.transform_to_R4(matrix)
    return BarycentricArc(new_start, new_end)