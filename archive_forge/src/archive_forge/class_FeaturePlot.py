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
class FeaturePlot(GeoPolygonPlot):
    scale = param.ObjectSelector(default='110m', objects=['10m', '50m', '110m'], doc='The scale of the Feature in meters.')

    def get_extents(self, element, ranges, range_type='combined', **kwargs):
        proj = self.projection
        if self.global_extent and range_type in ('combined', 'data'):
            (x0, x1), (y0, y1) = (proj.x_limits, proj.y_limits)
            return tuple((round(c, 12) for c in (x0, y0, x1, y1)))
        elif self.overlaid:
            return (np.nan,) * 4
        return super().get_extents(element, ranges, range_type)

    def get_data(self, element, ranges, style):
        mapping = dict(self._mapping)
        if self.static_source:
            return ({}, mapping, style)
        if hasattr(element.data, 'with_scale'):
            feature = element.data.with_scale(self.scale)
        else:
            feature = copy.copy(element.data)
            feature.scale = self.scale
        geoms = list(feature.geometries())
        if isinstance(geoms[0], line_types):
            el_type = Contours
            style['plot_method'] = 'multi_line'
            style.pop('fill_color', None)
            style.pop('fill_alpha', None)
        else:
            el_type = Polygons
        polys = el_type(geoms, crs=element.crs, **util.get_param_values(element))
        return super().get_data(polys, ranges, style)