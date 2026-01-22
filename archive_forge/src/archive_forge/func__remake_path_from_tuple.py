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
def _remake_path_from_tuple(props):
    """
    try to remake a path using the properties in props
    """
    if len(props) == 0:
        return ''

    def _add_square_brackets_to_number(n):
        if type(n) == type(int()):
            return '[%d]' % (n,)
        return n

    def _prepend_dot_if_not_number(s):
        if not s.startswith('['):
            return '.' + s
        return s
    props_all_str = list(map(_add_square_brackets_to_number, props))
    props_w_underscore = props_all_str[:1] + list(map(_prepend_dot_if_not_number, props_all_str[1:]))
    return ''.join(props_w_underscore)