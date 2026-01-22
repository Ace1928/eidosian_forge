import copy
import numpy as np
import param
import matplotlib.ticker as mticker
from cartopy import crs as ccrs
from cartopy.io.img_tiles import GoogleTiles, QuadtreeTiles
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from holoviews.core import Store, HoloMap, Layout, Overlay, Element, NdLayout
from holoviews.core import util
from holoviews.core.data import GridInterface
from holoviews.core.options import SkipRendering, Options
from holoviews.plotting.mpl import (
from holoviews.plotting.mpl.util import get_raster_array, wrap_formatter
from ...element import (
from ...util import geo_mesh, poly_types
from ..plot import ProjectionPlot
from ...operation import (
from .chart import WindBarbsPlot
class LayoutPlot(ProjectionPlot, HvLayoutPlot):
    """
    Extends HoloViews LayoutPlot with functionality to determine
    the correct projection for each axis.
    """
    vspace = param.Number(default=0.3, doc='\n      Specifies the space between vertically adjacent elements in the grid.\n      Default value is set conservatively to avoid overlap of subplots.')
    v17_layout_format = True