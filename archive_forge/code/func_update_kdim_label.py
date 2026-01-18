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
@app.callback(Output(component_id=kdim_label_id, component_property='children'), [Input(component_id=kdim_slider_id, component_property='value')])
def update_kdim_label(value, kdim_label=kdim_label):
    return f'{kdim_label}: {value:.2f}'