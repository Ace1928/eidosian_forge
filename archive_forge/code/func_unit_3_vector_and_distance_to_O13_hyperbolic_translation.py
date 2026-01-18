from snappy.SnapPy import matrix, vector
from snappy.hyperboloid import (r13_dot,
def unit_3_vector_and_distance_to_O13_hyperbolic_translation(v, d):
    """
    Takes a 3-vector in the unit tangent space at the origin of the
    hyperboloid model and a hyperbolic distance. Returns the
    O13-matrix corresponding to the translation moving the origin in
    the given direction by the given distance.
    """
    return unit_time_vector_to_o13_hyperbolic_translation([d.cosh()] + [d.sinh() * x for x in v])