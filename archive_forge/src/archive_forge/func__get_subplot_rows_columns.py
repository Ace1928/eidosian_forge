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
def _get_subplot_rows_columns(self):
    """
        Returns a pair of lists, the first containing all the row indices and
        the second all the column indices.
        """
    grid_ref = self._validate_get_grid_ref()
    nrows = len(grid_ref)
    ncols = len(grid_ref[0])
    return (range(1, nrows + 1), range(1, ncols + 1))