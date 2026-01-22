import warnings
from itertools import chain
from types import FunctionType
import bokeh
import bokeh.plotting
import numpy as np
import param
from bokeh.document.events import ModelChangedEvent
from bokeh.models import (
from bokeh.models.axes import CategoricalAxis, DatetimeAxis
from bokeh.models.formatters import (
from bokeh.models.layouts import TabPanel, Tabs
from bokeh.models.mappers import (
from bokeh.models.ranges import DataRange1d, FactorRange, Range1d
from bokeh.models.scales import LogScale
from bokeh.models.tickers import (
from bokeh.models.tools import Tool
from packaging.version import Version
from ...core import CompositeOverlay, Dataset, Dimension, DynamicMap, Element, util
from ...core.options import Keywords, SkipRendering, abbreviated_exception
from ...element import Annotation, Contours, Graph, Path, Tiles, VectorField
from ...streams import Buffer, PlotSize, RangeXY
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import color_intervals, dim_axis_label, dim_range_key, process_cmap
from .plot import BokehPlot
from .styles import (
from .tabular import TablePlot
from .util import (
def _get_factors(self, overlay, ranges):
    xfactors, yfactors = ([], [])
    for k, sp in self.subplots.items():
        el = overlay.data.get(k)
        if el is not None:
            elranges = util.match_spec(el, ranges)
            xfs, yfs = sp._get_factors(el, elranges)
            if len(xfs):
                xfactors.append(xfs)
            if len(yfs):
                yfactors.append(yfs)
    xfactors = list(util.unique_iterator(chain(*xfactors)))
    yfactors = list(util.unique_iterator(chain(*yfactors)))
    return (xfactors, yfactors)