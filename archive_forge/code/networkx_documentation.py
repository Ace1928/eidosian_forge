from collections import defaultdict
import numpy as np
import networkx as nx
import holoviews as _hv
from bokeh.models import HoverTool
from holoviews import Graph, Labels, dim
from holoviews.core.options import Store
from holoviews.core.util import dimension_sanitizer
from holoviews.plotting.bokeh import GraphPlot, LabelsPlot
from holoviews.plotting.bokeh.styles import markers
from .backend_transforms import _transfer_opts_cur_backend
from .util import process_crs
from .utilities import save, show # noqa
Draw networkx graph with planar layout.

    Parameters
    ----------
    G : graph
       A networkx graph
    kwargs : optional keywords
       See hvplot.networkx.draw() for a description of optional
       keywords, with the exception of the pos parameter which is not
       used by this function.

    Returns
    -------
    graph : holoviews.Graph or holoviews.Overlay
       Graph element or Graph and Labels
    