from math import cos, sin, tan, radians
def transformPoints(matrix, V):
    r = [transformPoint(matrix, v) for v in V]
    if isinstance(V, tuple):
        r = tuple(r)
    return r