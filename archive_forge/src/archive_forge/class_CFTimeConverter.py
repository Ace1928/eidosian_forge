import inspect
import re
import warnings
import matplotlib as mpl
import numpy as np
from matplotlib import (
from matplotlib.colors import Normalize, cnames
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Path, PathPatch
from matplotlib.rcsetup import validate_fontsize, validate_fonttype, validate_hatch
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from packaging.version import Version
from ...core.util import arraylike_types, cftime_types, is_number
from ...element import RGB, Polygons, Raster
from ..util import COLOR_ALIASES, RGB_HEX_REGEX
class CFTimeConverter(NetCDFTimeConverter):
    """
    Defines conversions for cftime types by extending nc_time_axis.
    """

    @classmethod
    def convert(cls, value, unit, axis):
        if not nc_axis_available:
            raise ValueError('In order to display cftime types with matplotlib install the nc_time_axis library using pip or from conda-forge using:\n\tconda install -c conda-forge nc_time_axis')
        if isinstance(value, cftime_types):
            value = CalendarDateTime(value.datetime, value.calendar)
        elif isinstance(value, np.ndarray):
            value = np.array([CalendarDateTime(v.datetime, v.calendar) for v in value])
        return super().convert(value, unit, axis)