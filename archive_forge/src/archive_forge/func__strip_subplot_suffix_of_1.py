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
def _strip_subplot_suffix_of_1(self, prop):
    """
        Strip the suffix for subplot property names that have a suffix of 1.
        All other properties are returned unchanged

        e.g. 'xaxis1' -> 'xaxis'

        Parameters
        ----------
        prop : str|tuple

        Returns
        -------
        str|tuple
        """
    prop_tuple = BaseFigure._str_to_dict_path(prop)
    if len(prop_tuple) != 1 or not isinstance(prop_tuple[0], str):
        return prop
    else:
        prop = prop_tuple[0]
    match = self._subplot_re_match(prop)
    if match:
        subplot_prop = match.group(1)
        suffix_digit = int(match.group(2))
        if subplot_prop and suffix_digit == 1:
            prop = subplot_prop
    return prop