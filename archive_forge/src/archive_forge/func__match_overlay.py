import warnings
import numpy as np
import param
from packaging.version import Version
from param import _is_number
from ..core import (
from ..core.data import ArrayInterface, DictInterface, PandasInterface, default_datatype
from ..core.data.util import dask_array_module
from ..core.util import (
from ..element.chart import Histogram, Scatter
from ..element.path import Contours, Polygons
from ..element.raster import RGB, Image
from ..element.util import categorical_aggregate2d  # noqa (API import)
from ..streams import RangeXY
from ..util.locator import MaxNLocator
def _match_overlay(self, raster, overlay_spec):
    """
        Given a raster or input overlay, generate a list of matched
        elements (None if no match) and corresponding tuple of match
        strength values.
        """
    ordering = [None] * len(overlay_spec)
    strengths = [0] * len(overlay_spec)
    elements = raster.values() if isinstance(raster, Overlay) else [raster]
    for el in elements:
        for pos in range(len(overlay_spec)):
            strength = self._match(el, overlay_spec[pos])
            if strength is None:
                continue
            elif strength <= strengths[pos]:
                continue
            else:
                ordering[pos] = el
                strengths[pos] = strength
    return (ordering, strengths)