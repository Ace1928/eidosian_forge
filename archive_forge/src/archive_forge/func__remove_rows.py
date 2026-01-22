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
def _remove_rows(self, rows=None):
    if rows is not None:
        selected_names = rows
    else:
        selected_names = list(map(lambda x: self._df.iloc[x].name, self._selected_rows))
    self._df.drop(selected_names, inplace=True)
    self._unfiltered_df.drop(selected_names, inplace=True)
    self._selected_rows = []
    self._update_table(triggered_by='remove_row')
    return selected_names