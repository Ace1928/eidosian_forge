import ipywidgets as widgets
import pandas as pd
import numpy as np
import json
from types import FunctionType
from IPython.display import display
from numbers import Integral
from traitlets import (
from itertools import chain
from uuid import uuid4
from six import string_types
from distutils.version import LooseVersion
def _get_col_series_from_df(self, col_name, df, level_vals=False):
    sort_column_name = self._sort_helper_columns.get(col_name)
    if sort_column_name:
        return df[sort_column_name]
    if col_name in self._primary_key:
        if len(self._primary_key) > 1:
            key_index = self._primary_key.index(col_name)
            if level_vals:
                return df.index.levels[key_index]
            return df.index.get_level_values(key_index)
        else:
            return df.index
    else:
        return df[col_name]