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
def _perform_select_traces(self, filter_by_subplot, grid_subplot_refs, selector):
    from plotly._subplots import _get_subplot_ref_for_trace

    def _filter_by_subplot_ref(trace):
        trace_subplot_ref = _get_subplot_ref_for_trace(trace)
        return trace_subplot_ref in grid_subplot_refs
    funcs = []
    if filter_by_subplot:
        funcs.append(_filter_by_subplot_ref)
    return _generator(self._filter_by_selector(self.data, funcs, selector))