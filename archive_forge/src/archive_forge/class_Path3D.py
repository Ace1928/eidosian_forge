import param
from ..core import Dimension, Element3D
from .geom import Points
from .path import Path
from .raster import Image
class Path3D(Element3D, Path):
    """
    Path3D is a 3D element representing a line through 3D space. The
    key dimensions represent the position of each coordinate along the
    x-, y- and z-axis while the value dimensions can optionally supply
    additional information.
    """
    kdims = param.List(default=[Dimension('x'), Dimension('y'), Dimension('z')], bounds=(3, 3))
    vdims = param.List(default=[], doc='\n        Path3D can have optional value dimensions.')
    group = param.String(default='Path3D', constant=True)

    def __getitem__(self, slc):
        return Path.__getitem__(self, slc)