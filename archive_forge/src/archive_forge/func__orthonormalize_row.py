from snappy.SnapPy import matrix, vector
from snappy.hyperboloid import (r13_dot,
def _orthonormalize_row(row, other_rows, row_sign):
    result = row
    for sign, other_row in zip(_signature, other_rows):
        s = sign * r13_dot(row, other_row)
        result = [c - s * other_c for c, other_c in zip(result, other_row)]
    try:
        result = R13_normalise(vector(result), sign=row_sign)
    except ValueError:
        return None
    if not _is_row_sane(result):
        return None
    return result