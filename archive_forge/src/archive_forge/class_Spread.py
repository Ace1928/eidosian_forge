import numpy as np
import param
from ..core import Dataset, Dimension, Element2D, NdOverlay, Overlay, util
from ..core.dimension import process_dimensions
from .geom import (  # noqa: F401 backward compatible import
from .selection import Selection1DExpr
class Spread(ErrorBars):
    """
    Spread is a Chart element representing a spread of values or
    confidence band in a 1D coordinate system. The key dimension(s)
    corresponds to the location along the x-axis and the value
    dimensions define the location along the y-axis as well as the
    symmetric or asymmetric spread.
    """
    group = param.String(default='Spread', constant=True)