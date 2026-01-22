import param
from ..core import Dimension, Element3D
from .geom import Points
from .path import Path
from .raster import Image
class TriSurface(Element3D, Points):
    """
    TriSurface represents a set of coordinates in 3D space which
    define a surface via a triangulation algorithm (usually Delauney
    triangulation). They key dimensions of a TriSurface define the
    position of each point along the x-, y- and z-axes, while value
    dimensions can provide additional information about each point.
    """
    group = param.String(default='TriSurface', constant=True)
    kdims = param.List(default=[Dimension('x'), Dimension('y'), Dimension('z')], bounds=(3, 3), doc='\n        The key dimensions of a TriSurface represent the 3D coordinates\n        of each point.')
    vdims = param.List(default=[], doc='\n        The value dimensions of a TriSurface can provide additional\n        information about each 3D coordinate.')

    def __getitem__(self, slc):
        return Points.__getitem__(self, slc)