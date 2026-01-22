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
def _set_in(d, key_path_str, v):
    """
        Set a value in a nested dict using a key path string
        (e.g. 'foo.bar[0]')

        Parameters
        ----------
        d : dict
            Input dict to set property in
        key_path_str : str
            Key path string, where nested keys are joined on '.' characters
            and array indexes are specified using brackets
            (e.g. 'foo.bar[1]')
        v
            New value
        Returns
        -------
        bool
            True if set resulted in modification of dict (i.e. v was not
            already present at the specified location), False otherwise.
        """
    assert isinstance(d, dict)
    key_path = BaseFigure._str_to_dict_path(key_path_str)
    val_parent = d
    for kp, key_path_el in enumerate(key_path[:-1]):
        if isinstance(val_parent, list) and isinstance(key_path_el, int):
            while len(val_parent) <= key_path_el:
                val_parent.append(None)
        elif isinstance(val_parent, dict) and key_path_el not in val_parent:
            if isinstance(key_path[kp + 1], int):
                val_parent[key_path_el] = []
            else:
                val_parent[key_path_el] = {}
        val_parent = val_parent[key_path_el]
    last_key = key_path[-1]
    val_changed = False
    if v is Undefined:
        pass
    elif v is None:
        if isinstance(val_parent, dict):
            if last_key in val_parent:
                val_parent.pop(last_key)
                val_changed = True
        elif isinstance(val_parent, list):
            if isinstance(last_key, int) and 0 <= last_key < len(val_parent):
                val_parent[last_key] = None
                val_changed = True
        else:
            raise ValueError('\n    Cannot remove element of type {typ} at location {raw_key}'.format(typ=type(val_parent), raw_key=key_path_str))
    elif isinstance(val_parent, dict):
        if last_key not in val_parent or not BasePlotlyType._vals_equal(val_parent[last_key], v):
            val_parent[last_key] = v
            val_changed = True
    elif isinstance(val_parent, list):
        if isinstance(last_key, int):
            while len(val_parent) <= last_key:
                val_parent.append(None)
            if not BasePlotlyType._vals_equal(val_parent[last_key], v):
                val_parent[last_key] = v
                val_changed = True
    else:
        raise ValueError('\n    Cannot set element of type {typ} at location {raw_key}'.format(typ=type(val_parent), raw_key=key_path_str))
    return val_changed