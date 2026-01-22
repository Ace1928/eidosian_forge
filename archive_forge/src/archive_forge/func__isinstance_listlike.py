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
def _isinstance_listlike(x):
    """Returns True if x is an iterable which can be transformed into a pandas Series,
    False for the other types of possible values of a `hover_data` dict.
    A tuple of length 2 is a special case corresponding to a (format, data) tuple.
    """
    if isinstance(x, str) or (isinstance(x, tuple) and len(x) == 2) or isinstance(x, bool) or (x is None):
        return False
    else:
        return True