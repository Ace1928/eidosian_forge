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
def _perform_plotly_update(self, restyle_data=None, relayout_data=None, trace_indexes=None):
    if not restyle_data and (not relayout_data):
        return (None, None, None)
    if restyle_data is None:
        restyle_data = {}
    if relayout_data is None:
        relayout_data = {}
    trace_indexes = self._normalize_trace_indexes(trace_indexes)
    relayout_changes = self._perform_plotly_relayout(relayout_data)
    restyle_changes = self._perform_plotly_restyle(restyle_data, trace_indexes)
    return (restyle_changes, relayout_changes, trace_indexes)