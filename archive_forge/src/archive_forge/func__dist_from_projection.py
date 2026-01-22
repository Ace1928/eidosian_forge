from snappy.SnapPy import matrix, vector
from snappy.hyperboloid import (r13_dot,
def _dist_from_projection(p, dir):
    return (p / dir).imag() * abs(dir)