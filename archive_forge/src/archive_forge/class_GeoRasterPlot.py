import copy
import param
import numpy as np
from cartopy.crs import GOOGLE_MERCATOR
from bokeh.models import WMTSTileSource, BBoxTileSource, QUADKEYTileSource, SaveTool
from holoviews import Store, Overlay, NdOverlay
from holoviews.core import util
from holoviews.core.options import SkipRendering, Options, Compositor
from holoviews.plotting.bokeh.annotation import TextPlot, LabelsPlot
from holoviews.plotting.bokeh.chart import PointPlot, VectorFieldPlot
from holoviews.plotting.bokeh.geometry import RectanglesPlot, SegmentPlot
from holoviews.plotting.bokeh.graphs import TriMeshPlot, GraphPlot
from holoviews.plotting.bokeh.hex_tiles import hex_binning, HexTilesPlot
from holoviews.plotting.bokeh.path import PolygonPlot, PathPlot, ContourPlot
from holoviews.plotting.bokeh.raster import RasterPlot, RGBPlot, QuadMeshPlot
from ...element import (
from ...operation import (
from ...tile_sources import _ATTRIBUTIONS
from ...util import poly_types, line_types
from .plot import GeoPlot, GeoOverlayPlot
from . import callbacks # noqa
class GeoRasterPlot(GeoPlot, RasterPlot):
    clipping_colors = param.Dict(default={'NaN': (0, 0, 0, 0)}, doc='\n        Dictionary to specify colors for clipped values, allows\n        setting color for NaN values and for values above and below\n        the min and max value. The min, max or NaN color may specify\n        an RGB(A) color as a color hex string of the form #FFFFFF or\n        #FFFFFFFF or a length 3 or length 4 tuple specifying values in\n        the range 0-1 or a named HTML color.')
    _project_operation = project_image.instance(fast=False)
    _hover_code = '\n        var projections = Bokeh.require("core/util/projections");\n        var x = special_vars.x\n        var y = special_vars.y\n        var coords = projections.wgs84_mercator.invert(x, y)\n        return "" + (coords[%d]).toFixed(4)\n    '