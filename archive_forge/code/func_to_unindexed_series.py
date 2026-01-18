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
def to_unindexed_series(x, name=None):
    """
    assuming x is list-like or even an existing pd.Series, return a new pd.Series with
    no index, without extracting the data from an existing Series via numpy, which
    seems to mangle datetime columns. Stripping the index from existing pd.Series is
    required to get things to match up right in the new DataFrame we're building
    """
    return pd.Series(x, name=name).reset_index(drop=True)