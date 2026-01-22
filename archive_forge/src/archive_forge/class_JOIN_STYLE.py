import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
class JOIN_STYLE:
    round = BufferJoinStyle.round
    mitre = BufferJoinStyle.mitre
    bevel = BufferJoinStyle.bevel