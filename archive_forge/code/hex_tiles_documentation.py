from collections.abc import Callable
import numpy as np
import param
from bokeh.util.hex import cartesian_to_axial
from ...core import Dimension, Operation
from ...core.options import Compositor
from ...core.util import isfinite, max_range
from ...element import HexTiles
from ...util.transform import dim as dim_transform
from .element import ColorbarPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties

    Applies hex binning by computing aggregates on a hexagonal grid.

    Should not be user facing as the returned element is not directly
    usable.
    