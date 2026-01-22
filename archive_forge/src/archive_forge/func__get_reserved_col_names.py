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
def _get_reserved_col_names(args):
    """
    This function builds a list of columns of the data_frame argument used
    as arguments, either as str/int arguments or given as columns
    (pandas series type).
    """
    df = args['data_frame']
    reserved_names = set()
    for field in args:
        if field not in all_attrables:
            continue
        names = args[field] if field in array_attrables else [args[field]]
        if names is None:
            continue
        for arg in names:
            if arg is None:
                continue
            elif isinstance(arg, str):
                reserved_names.add(arg)
            elif isinstance(arg, pd.Series):
                arg_name = arg.name
                if arg_name and hasattr(df, arg_name):
                    in_df = arg is df[arg_name]
                    if in_df:
                        reserved_names.add(arg_name)
            elif arg is df.index and arg.name is not None:
                reserved_names.add(arg.name)
    return reserved_names