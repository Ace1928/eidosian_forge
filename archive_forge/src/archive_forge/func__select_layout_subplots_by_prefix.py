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
def _select_layout_subplots_by_prefix(self, prefix, selector=None, row=None, col=None, secondary_y=None):
    """
        Helper called by code generated select_* methods
        """
    if row is not None or col is not None or secondary_y is not None:
        grid_ref = self._validate_get_grid_ref()
        container_to_row_col = {}
        for r, subplot_row in enumerate(grid_ref):
            for c, subplot_refs in enumerate(subplot_row):
                if not subplot_refs:
                    continue
                for i, subplot_ref in enumerate(subplot_refs):
                    for layout_key in subplot_ref.layout_keys:
                        if layout_key.startswith(prefix):
                            is_secondary_y = i == 1
                            container_to_row_col[layout_key] = (r + 1, c + 1, is_secondary_y)
    else:
        container_to_row_col = None
    layout_keys_filters = [lambda k: k.startswith(prefix) and self.layout[k] is not None, lambda k: row is None or container_to_row_col.get(k, (None, None, None))[0] == row, lambda k: col is None or container_to_row_col.get(k, (None, None, None))[1] == col, lambda k: secondary_y is None or container_to_row_col.get(k, (None, None, None))[2] == secondary_y]
    layout_keys = reduce(lambda last, f: filter(f, last), layout_keys_filters, _natural_sort_strings(list(self.layout)))
    layout_objs = [self.layout[k] for k in layout_keys]
    return _generator(self._filter_by_selector(layout_objs, [], selector))