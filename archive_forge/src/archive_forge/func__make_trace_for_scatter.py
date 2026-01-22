from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.subplots import make_subplots
import math
from numbers import Number
def _make_trace_for_scatter(trace, trace_type, color, **kwargs_marker):
    if trace_type in ['scatter', 'scattergl']:
        trace['mode'] = 'markers'
        trace['marker'] = dict(color=color, **kwargs_marker)
    return trace