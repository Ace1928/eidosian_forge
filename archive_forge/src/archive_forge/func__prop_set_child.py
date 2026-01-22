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
def _prop_set_child(self, child, prop_path_str, val):
    """
        Propagate property setting notification from child to parent

        Parameters
        ----------
        child : BasePlotlyType
            Child object
        prop_path_str : str
            Property path string (e.g. 'foo[0].bar') of property that
            was set, relative to `child`
        val
            New value for property. Either a simple value, a dict,
            or a tuple of dicts. This should *not* be a BasePlotlyType object.

        Returns
        -------
        None
        """
    child_prop_val = getattr(self, child.plotly_name)
    if isinstance(child_prop_val, (list, tuple)):
        child_ind = BaseFigure._index_is(child_prop_val, child)
        obj_path = '{child_name}.{child_ind}.{prop}'.format(child_name=child.plotly_name, child_ind=child_ind, prop=prop_path_str)
    else:
        obj_path = '{child_name}.{prop}'.format(child_name=child.plotly_name, prop=prop_path_str)
    self._send_prop_set(obj_path, val)