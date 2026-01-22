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
@staticmethod
def _selector_matches(obj, selector):
    if selector is None:
        return True
    if isinstance(selector, str):
        selector = dict(type=selector)
    if isinstance(selector, dict) or isinstance(selector, BasePlotlyType):
        for k in selector:
            if k not in obj:
                return False
            obj_val = obj[k]
            selector_val = selector[k]
            if isinstance(obj_val, BasePlotlyType):
                obj_val = obj_val.to_plotly_json()
            if isinstance(selector_val, BasePlotlyType):
                selector_val = selector_val.to_plotly_json()
            if obj_val != selector_val:
                return False
        return True
    elif callable(selector):
        return selector(obj)
    else:
        raise TypeError('selector must be dict or a function accepting a graph object returning a boolean.')