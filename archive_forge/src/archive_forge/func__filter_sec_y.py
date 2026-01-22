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
def _filter_sec_y(obj):
    """Filter objects on secondary y axes"""
    return secondary_y is None or yref_to_secondary_y.get(obj.yref, None) == secondary_y