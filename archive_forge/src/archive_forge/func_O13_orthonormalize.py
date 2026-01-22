from snappy.SnapPy import matrix, vector
from snappy.hyperboloid import (r13_dot,
def O13_orthonormalize(m):
    try:
        ring = m[0][0].parent()
    except AttributeError:
        ring = None
    id_matrix = matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], ring=ring)
    result = []
    for row, id_row, sign in zip(m, id_matrix, _signature):
        result.append(_orthonormalize_row_sane(row, id_row, result, sign))
    return matrix(result, ring=ring)