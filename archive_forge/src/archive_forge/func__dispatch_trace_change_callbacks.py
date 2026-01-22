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
def _dispatch_trace_change_callbacks(self, restyle_data, trace_indexes):
    """
        Dispatch property change callbacks given restyle_data

        Parameters
        ----------
        restyle_data : dict[str, any]
            See docstring for plotly_restyle.

        trace_indexes : list[int]
            List of trace indexes that restyle operation applied to

        Returns
        -------
        None
        """
    key_path_strs = list(restyle_data.keys())
    dispatch_plan = BaseFigure._build_dispatch_plan(key_path_strs)
    for path_tuple, changed_paths in dispatch_plan.items():
        for trace_ind in trace_indexes:
            trace = self.data[trace_ind]
            if path_tuple in trace:
                dispatch_obj = trace[path_tuple]
                if isinstance(dispatch_obj, BasePlotlyType):
                    dispatch_obj._dispatch_change_callbacks(changed_paths)