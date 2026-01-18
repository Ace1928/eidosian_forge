import base64
import copy
import pickle
import uuid
from collections import namedtuple
from dash.exceptions import PreventUpdate
import holoviews as hv
from holoviews.core.decollate import (
from holoviews.plotting.plotly import DynamicMap, PlotlyRenderer
from holoviews.plotting.plotly.callbacks import (
from holoviews.plotting.plotly.util import clean_internal_figure_properties
from holoviews.streams import Derived, History
import plotly.graph_objects as go
from dash import callback_context
from dash.dependencies import Input, Output, State
def to_function_spec(hvobj):
    """
    Convert Dynamic HoloViews object into a pure function that accepts kdim values
    and stream contents as positional arguments.

    This borrows the low-level holoviews decollate logic, but instead of returning
    DynamicMap with cloned streams, returns a HoloViewsFunctionSpec.

    Args:
        hvobj: A potentially dynamic Holoviews object

    Returns:
        HoloViewsFunctionSpec
    """
    kdims_list = []
    original_streams = []
    streams = []
    stream_mapping = {}
    initialize_dynamic(hvobj)
    expr = to_expr_extract_streams(hvobj, kdims_list, streams, original_streams, stream_mapping)
    expr_fn = expr_to_fn_of_stream_contents(expr, nkdims=len(kdims_list))
    if isinstance(hvobj, DynamicMap) and hvobj.unbounded:
        dims = ', '.join((f'{dim!r}' for dim in hvobj.unbounded))
        msg = 'DynamicMap cannot be displayed without explicit indexing as {dims} dimension(s) are unbounded. \nSet dimensions bounds with the DynamicMap redim.range or redim.values methods.'
        raise ValueError(msg.format(dims=dims))
    dimensions_dict = {d.name: d for d in hvobj.dimensions()}
    kdims = {}
    for k in kdims_list:
        dim = dimensions_dict[k.name]
        label = dim.label or dim.name
        kdims[k.name] = (label, dim.values or dim.range)
    return HoloViewsFunctionSpec(fn=expr_fn, kdims=kdims, streams=original_streams)