from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.subplots import make_subplots
import math
from numbers import Number
def _axis_title_annotation(text, x_or_y_axis):
    if x_or_y_axis == 'x':
        x_pos = 0.5
        y_pos = -0.1
        textangle = 0
    elif x_or_y_axis == 'y':
        x_pos = -0.1
        y_pos = 0.5
        textangle = 270
    if not text:
        text = ''
    annot = {'font': {'color': '#000000', 'size': AXIS_TITLE_SIZE}, 'showarrow': False, 'text': text, 'textangle': textangle, 'x': x_pos, 'xanchor': 'center', 'xref': 'paper', 'y': y_pos, 'yanchor': 'middle', 'yref': 'paper'}
    return annot