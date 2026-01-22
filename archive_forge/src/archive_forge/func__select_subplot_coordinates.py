import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
def _select_subplot_coordinates(self, rows, cols, product=False):
    """
        Allows selecting all or a subset of the subplots.
        If any of rows or columns is 'all', product is set to True. This is
        probably the expected behaviour, so that rows=1,cols='all' selects all
        the columns in row 1 (otherwise it would just select the subplot in the
        first row and first column).
        """
    product |= any([s == 'all' for s in [rows, cols]])
    t = _indexing_combinations([rows, cols], list(self._get_subplot_rows_columns()), product=product)
    t = list(t)
    grid_ref = self._validate_get_grid_ref()
    t = list(filter(lambda u: grid_ref[u[0] - 1][u[1] - 1] is not None, t))
    return t