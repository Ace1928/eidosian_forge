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
def _filter_by_subplot_ref(trace):
    trace_subplot_ref = _get_subplot_ref_for_trace(trace)
    return trace_subplot_ref in grid_subplot_refs