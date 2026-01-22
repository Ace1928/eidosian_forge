import numpy as np
import param
from ..core import Dataset, Dimension, Element2D, NdOverlay, Overlay, util
from ..core.dimension import process_dimensions
from .geom import (  # noqa: F401 backward compatible import
from .selection import Selection1DExpr
class Spikes(Selection1DExpr, Chart):
    """
    Spikes is a Chart element which represents a number of discrete
    spikes, events or observations in a 1D coordinate system. The key
    dimension therefore represents the position of each spike along
    the x-axis while the first value dimension, if defined, controls
    the height along the y-axis. It may therefore be used to visualize
    the distribution of discrete events, representing a rug plot, or
    to draw the strength some signal.
    """
    group = param.String(default='Spikes', constant=True)
    kdims = param.List(default=[Dimension('x')], bounds=(1, 1))
    vdims = param.List(default=[], bounds=(0, None))
    _auto_indexable_1d = False