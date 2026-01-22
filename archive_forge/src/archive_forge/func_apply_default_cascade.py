import plotly.graph_objs as go
import plotly.io as pio
from collections import namedtuple, OrderedDict
from ._special_inputs import IdentityMap, Constant, Range
from .trendline_functions import ols, lowess, rolling, expanding, ewm
from _plotly_utils.basevalidators import ColorscaleValidator
from plotly.colors import qualitative, sequential
import math
from packaging import version
import pandas as pd
import numpy as np
from plotly._subplots import (
def apply_default_cascade(args):
    for param in defaults.__slots__:
        if param in args and args[param] is None:
            args[param] = getattr(defaults, param)
    if args['template'] is None:
        if pio.templates.default is not None:
            args['template'] = pio.templates.default
        else:
            args['template'] = 'plotly'
    try:
        args['template'] = pio.templates[args['template']]
    except Exception:
        args['template'] = go.layout.Template(args['template'])
    if 'color_continuous_scale' in args:
        if args['color_continuous_scale'] is None and args['template'].layout.colorscale.sequential:
            args['color_continuous_scale'] = [x[1] for x in args['template'].layout.colorscale.sequential]
        if args['color_continuous_scale'] is None:
            args['color_continuous_scale'] = sequential.Viridis
    if 'color_discrete_sequence' in args:
        if args['color_discrete_sequence'] is None and args['template'].layout.colorway:
            args['color_discrete_sequence'] = args['template'].layout.colorway
        if args['color_discrete_sequence'] is None:
            args['color_discrete_sequence'] = qualitative.D3
    if 'symbol_sequence' in args:
        if args['symbol_sequence'] is None and args['template'].data.scatter:
            args['symbol_sequence'] = [scatter.marker.symbol for scatter in args['template'].data.scatter]
        if not args['symbol_sequence'] or not any(args['symbol_sequence']):
            args['symbol_sequence'] = ['circle', 'diamond', 'square', 'x', 'cross']
    if 'line_dash_sequence' in args:
        if args['line_dash_sequence'] is None and args['template'].data.scatter:
            args['line_dash_sequence'] = [scatter.line.dash for scatter in args['template'].data.scatter]
        if not args['line_dash_sequence'] or not any(args['line_dash_sequence']):
            args['line_dash_sequence'] = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
    if 'pattern_shape_sequence' in args:
        if args['pattern_shape_sequence'] is None and args['template'].data.bar:
            args['pattern_shape_sequence'] = [bar.marker.pattern.shape for bar in args['template'].data.bar]
        if not args['pattern_shape_sequence'] or not any(args['pattern_shape_sequence']):
            args['pattern_shape_sequence'] = ['', '/', '\\', 'x', '+', '.']