import math
import numpy
from rdkit import Chem, Geometry
def ArbAxisRotation(theta, ax, pt):
    theta = math.pi * theta / 180
    c = math.cos(theta)
    s = math.sin(theta)
    t = 1 - c
    X = ax.x
    Y = ax.y
    Z = ax.z
    mat = [[t * X * X + c, t * X * Y + s * Z, t * X * Z - s * Y], [t * X * Y - s * Z, t * Y * Y + c, t * Y * Z + s * X], [t * X * Z + s * Y, t * Y * Z - s * X, t * Z * Z + c]]
    mat = numpy.array(mat, dtype=numpy.float64)
    if isinstance(pt, Geometry.Point3D):
        pt = numpy.array((pt.x, pt.y, pt.z))
        tmp = numpy.dot(mat, pt)
        return Geometry.Point3D(tmp[0], tmp[1], tmp[2])
    if isinstance(pt, list) or isinstance(pt, tuple):
        res = []
        for p in pt:
            tmp = numpy.dot(mat, numpy.array((p.x, p.y, p.z)))
            res.append(Geometry.Point3D(tmp[0], tmp[1], tmp[2]))
        return res
    return None