import snappy
import regina
import snappy.snap.t3mlite as t3m
import snappy.snap.t3mlite.spun as spun
def vertex_surfaces(regina_triangulation):
    """
    Enumerate the vertex surfaces of the given triangulation
    in quad coordinates.
    """
    surfaces = regina.NNormalSurfaceList.enumerate(regina_triangulation, regina.NS_QUAD)
    for i in range(surfaces.getNumberOfSurfaces()):
        yield surfaces.getSurface(i)