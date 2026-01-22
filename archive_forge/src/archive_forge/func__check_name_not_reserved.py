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
def _check_name_not_reserved(field_name, reserved_names):
    if field_name not in reserved_names:
        return field_name
    else:
        raise NameError("A name conflict was encountered for argument '%s'. A column or index with name '%s' is ambiguous." % (field_name, field_name))