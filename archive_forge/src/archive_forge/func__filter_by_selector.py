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
def _filter_by_selector(self, objects, funcs, selector):
    """
        objects is a sequence of objects, funcs a list of functions that
        return True if the object should be included in the selection and False
        otherwise and selector is an argument to the self._selector_matches
        function.
        If selector is an integer, the resulting sequence obtained after
        sucessively filtering by each function in funcs is indexed by this
        integer.
        Otherwise selector is used as the selector argument to
        self._selector_matches which is used to filter down the sequence.
        The function returns the sequence (an iterator).
        """
    if not isinstance(selector, int):
        funcs.append(lambda obj: self._selector_matches(obj, selector))

    def _filt(last, f):
        return filter(f, last)
    filtered_objects = reduce(_filt, funcs, objects)
    if isinstance(selector, int):
        return iter([list(filtered_objects)[selector]])
    return filtered_objects