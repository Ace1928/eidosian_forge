from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.subplots import make_subplots
import math
from numbers import Number
def _is_flipped(num):
    if num >= THRES_FOR_FLIPPED_FACET_TITLES:
        flipped = True
    else:
        flipped = False
    return flipped