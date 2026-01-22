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
def _perform_plotly_restyle(self, restyle_data, trace_indexes):
    """
        Perform a restyle operation on the figure's traces data and return
        the changes that were applied

        Parameters
        ----------
        restyle_data : dict[str, any]
            See docstring for plotly_restyle
        trace_indexes : list[int]
            List of trace indexes that restyle operation applies to
        Returns
        -------
        restyle_changes: dict[str, any]
            Subset of restyle_data including only the keys / values that
            resulted in a change to the figure's traces data
        """
    restyle_changes = {}
    for key_path_str, v in restyle_data.items():
        any_vals_changed = False
        for i, trace_ind in enumerate(trace_indexes):
            if trace_ind >= len(self._data):
                raise ValueError('Trace index {trace_ind} out of range'.format(trace_ind=trace_ind))
            trace_v = v[i % len(v)] if isinstance(v, list) else v
            if trace_v is not Undefined:
                trace_obj = self.data[trace_ind]
                if not BaseFigure._is_key_path_compatible(key_path_str, trace_obj):
                    trace_class = trace_obj.__class__.__name__
                    raise ValueError("\nInvalid property path '{key_path_str}' for trace class {trace_class}\n".format(key_path_str=key_path_str, trace_class=trace_class))
                val_changed = BaseFigure._set_in(self._data[trace_ind], key_path_str, trace_v)
                any_vals_changed = any_vals_changed or val_changed
        if any_vals_changed:
            restyle_changes[key_path_str] = v
    return restyle_changes