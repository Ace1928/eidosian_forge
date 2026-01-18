from math import cos, sin, tan, radians
def zTransformPoint(A, v):
    """Apply the homogenous part of atransformation a to vector v --> A*v"""
    return (A[0] * v[0] + A[2] * v[1], A[1] * v[0] + A[3] * v[1])