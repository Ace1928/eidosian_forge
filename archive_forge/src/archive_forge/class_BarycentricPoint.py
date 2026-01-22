from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
class BarycentricPoint(Point):
    """
    A quadruple of Sage rational numbers whose sum is 1.
    """

    def __init__(self, c0, c1, c2, c3):
        self.vector = Vector4([c0, c1, c2, c3])
        self._zero_coordinates = [i for i in range(4) if self.vector[i] == 0]
        if sum(self.vector) != 1:
            raise Exception("Barycentric point doesn't sum to 1")

    def __repr__(self):
        return self.vector.__repr__()

    def __eq__(self, other):
        return self.vector == other.vector

    def __ne__(self, other):
        return self.vector != other.vector

    def __hash__(self):
        return hash(self.vector)

    def has_negative_coordinate(self):
        for l in self.vector:
            if l < 0:
                return True
        return False

    def negative_coordinates(self):
        return [l for l in self.vector if l < 0]

    def zero_coordinates(self):
        return self._zero_coordinates

    def on_boundary(self):
        return len(self._zero_coordinates) > 0

    def boundary_face(self):
        zeros = self._zero_coordinates
        if len(zeros) != 1:
            raise GeneralPositionError('Not a generic point on a face')
        return zeros[0]

    def is_interior(self):
        return len(self._zero_coordinates) == 0

    def convex_combination(self, other, t):
        c0, c1, c2, c3 = (1 - t) * self.vector + t * other.vector
        return BarycentricPoint(c0, c1, c2, c3)

    def transform_to_R3(self, matrix, bdry_map=None):
        v = self.vector
        new_v = matrix * v
        boundary_face = None
        if bdry_map is not None and self.on_boundary():
            face = self.zero_coordinates()[0]
            boundary_face = bdry_map[face]
        return Point(new_v[0], new_v[1], new_v[2], boundary_face)

    def min_nonzero(self):
        return min([c for c in self.vector if c > 0])

    def to_3d_point(self):
        return self.vector[0:3]

    def permute(self, perm):
        """
        Start with a permutation perm, which represents a map from the
        vertices of a tetrahedron T in which a point lies, to the
        vertices of another tetrahedron S which is glued to T. Then,
        translate the barycentric coordinates of the point in T to the
        corresponding barycentric coordinates of the point in S.  On
        the interior, this doesn't make much sense; it really should
        be used for points on the common boundary triangle.
        """
        v = self.vector
        new_v = [0] * 4
        for i in range(4):
            new_v[perm[i]] = v[i]
        return BarycentricPoint(*new_v)

    def round(self, max_denom=2 ** 32, force=False):
        if force or max((x.denominator() for x in self.vector)) > max_denom:
            v = []
            for y in max_denom * self.vector:
                if y != 0:
                    y = max(y.round(), 1)
                v.append(y)
            self.vector = Vector4(Vector4(v) / sum(v))
            self._zero_coordinates = [i for i in range(4) if self.vector[i] == 0]