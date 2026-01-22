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
def _duplicate_last_row(self):
    """
        Append a row at the end of the DataFrame by duplicating the
        last row and incrementing it's index by 1. The method is only
        available for DataFrames that have an integer index.
        """
    df = self._df
    if not df.index.is_integer():
        msg = 'Cannot add a row to a table with a non-integer index'
        self.send({'type': 'show_error', 'error_msg': msg, 'triggered_by': 'add_row'})
        return
    last_index = max(df.index)
    last = df.loc[last_index].copy()
    last.name += 1
    last[self._index_col_name] = last.name
    df.loc[last.name] = last.values
    self._unfiltered_df.loc[last.name] = last.values
    self._update_table(triggered_by='add_row', scroll_to_row=df.index.get_loc(last.name))
    return last.name