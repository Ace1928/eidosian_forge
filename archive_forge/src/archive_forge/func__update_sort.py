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
def _update_sort(self):
    try:
        if self._sort_field is None:
            return
        self._disable_grouping = False
        if self._sort_field in self._primary_key:
            if len(self._primary_key) == 1:
                self._df.sort_index(ascending=self._sort_ascending, inplace=True)
            else:
                level_index = self._primary_key.index(self._sort_field)
                self._df.sort_index(level=level_index, ascending=self._sort_ascending, inplace=True)
                if level_index > 0:
                    self._disable_grouping = True
        else:
            self._df.sort_values(self._sort_field, ascending=self._sort_ascending, inplace=True)
            self._disable_grouping = True
    except TypeError:
        self.log.info('TypeError occurred, assuming mixed data type column')
        self._df.sort_values(self._initialize_sort_column(self._sort_field), ascending=self._sort_ascending, inplace=True)