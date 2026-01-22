from sympy.core import Basic, diff
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.matrices import Matrix
from sympy.integrals import Integral, integrate
from sympy.geometry.entity import GeometryEntity
from sympy.simplify.simplify import simplify
from sympy.utilities.iterables import topological_sort
from sympy.vector import (CoordSys3D, Vector, ParametricRegion,
from sympy.vector.operators import _get_coord_systems
@classmethod
def _bounds_case(cls, parameters, limits):
    V = list(limits.keys())
    E = []
    for p in V:
        lower_p = limits[p][0]
        upper_p = limits[p][1]
        lower_p = lower_p.atoms()
        upper_p = upper_p.atoms()
        E.extend(((p, q) for q in V if p != q and (lower_p.issuperset({q}) or upper_p.issuperset({q}))))
    if not E:
        return parameters
    else:
        return topological_sort((V, E), key=default_sort_key)