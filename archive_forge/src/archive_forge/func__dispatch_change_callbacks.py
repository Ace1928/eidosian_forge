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
def _dispatch_change_callbacks(self, changed_paths):
    """
        Execute the appropriate change callback functions given a set of
        changed property path tuples

        Parameters
        ----------
        changed_paths : set[tuple[int|str]]

        Returns
        -------
        None
        """
    for prop_path_tuples, callbacks in self._change_callbacks.items():
        common_paths = changed_paths.intersection(set(prop_path_tuples))
        if common_paths:
            callback_args = [self[cb_path] for cb_path in prop_path_tuples]
            for callback in callbacks:
                callback(self, *callback_args)