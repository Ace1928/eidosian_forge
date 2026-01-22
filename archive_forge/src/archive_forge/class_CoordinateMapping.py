from __future__ import annotations
import logging # isort:skip
from ..core.properties import Instance, InstanceDefault
from ..model import Model
from .ranges import DataRange1d, Range
from .scales import LinearScale, Scale
class CoordinateMapping(Model):
    """ A mapping between two coordinate systems. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    x_source = Instance(Range, default=InstanceDefault(DataRange1d), help='\n    The source range of the horizontal dimension of the new coordinate space.\n    ')
    y_source = Instance(Range, default=InstanceDefault(DataRange1d), help='\n    The source range of the vertical dimension of the new coordinate space.\n    ')
    x_scale = Instance(Scale, default=InstanceDefault(LinearScale), help='\n    What kind of scale to use to convert x-coordinates from the source (data)\n    space into x-coordinates in the target (possibly screen) coordinate space.\n    ')
    y_scale = Instance(Scale, default=InstanceDefault(LinearScale), help='\n    What kind of scale to use to convert y-coordinates from the source (data)\n    space into y-coordinates in the target (possibly screen) coordinate space.\n    ')
    x_target = Instance(Range, help='\n    The horizontal range to map x-coordinates in the target coordinate space.\n    ')
    y_target = Instance(Range, help='\n    The vertical range to map y-coordinates in the target coordinate space.\n    ')