from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def arc_distance(arc_a, arc_b):
    return rational_sqrt(arc_distance_sq(arc_a, arc_b))