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
def _make_hyphen_key(key):
    if '_' in key[1:]:
        for under_prop, hyphen_prop in underscore_props.items():
            key = key.replace(under_prop, hyphen_prop)
    return key