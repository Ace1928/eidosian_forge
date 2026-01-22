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
def _to_ordered_dict(d, skip_uid=False):
    """
        Static helper for converting dict or list to structure of ordered
        dictionaries
        """
    if isinstance(d, dict):
        result = collections.OrderedDict()
        for key in sorted(d.keys()):
            if skip_uid and key == 'uid':
                continue
            else:
                result[key] = BaseFigure._to_ordered_dict(d[key], skip_uid=skip_uid)
    elif isinstance(d, list) and d and isinstance(d[0], dict):
        result = [BaseFigure._to_ordered_dict(el, skip_uid=skip_uid) for el in d]
    else:
        result = d
    return result