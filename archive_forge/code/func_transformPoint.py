from math import cos, sin, tan, radians
def transformPoint(A, v):
    """Apply transformation a to vector v --> A*v"""
    return (A[0] * v[0] + A[2] * v[1] + A[4], A[1] * v[0] + A[3] * v[1] + A[5])