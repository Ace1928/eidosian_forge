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
def _len_dict_item(item):
    """
    Because a parsed dict path is a tuple containings strings or integers, to
    know the length of the resulting string when printing we might need to
    convert to a string before calling len on it.
    """
    try:
        l = len(item)
    except TypeError:
        try:
            l = len('%d' % (item,))
        except TypeError:
            raise ValueError('Cannot find string length of an item that is not string-like nor an integer.')
    return l