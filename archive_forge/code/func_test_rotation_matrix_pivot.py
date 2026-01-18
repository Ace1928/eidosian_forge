import math
from affine import Affine
def test_rotation_matrix_pivot():
    """A rotation matrix with pivot has expected elements"""
    rot = Affine.rotation(90.0, pivot=(1.0, 1.0))
    exp = Affine.translation(1.0, 1.0) * Affine.rotation(90.0) * Affine.translation(-1.0, -1.0)
    for r, e in zip(rot, exp):
        assert round(r, 15) == round(e, 15)