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
def update_stream_values_for_type(store_data, stream_event_data, uid_to_streams_for_type):
    """
    Update the store with values of streams for a single type

    Args:
        store_data: Current store dictionary
        stream_event_data:  Potential stream data for current plotly event and
            traces in figures
        uid_to_streams_for_type: Mapping from trace UIDs to HoloViews streams of
            a particular type
    Returns:
        any_change: Whether any stream value has been updated
    """
    any_change = False
    for uid, event_data in stream_event_data.items():
        if uid in uid_to_streams_for_type:
            for stream_id in uid_to_streams_for_type[uid]:
                if stream_id not in store_data['streams'] or store_data['streams'][stream_id] != event_data:
                    store_data['streams'][stream_id] = event_data
                    any_change = True
    return any_change