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
def _perform_update(plotly_obj, update_obj, overwrite=False):
    """
        Helper to support the update() methods on :class:`BaseFigure` and
        :class:`BasePlotlyType`

        Parameters
        ----------
        plotly_obj : BasePlotlyType|tuple[BasePlotlyType]
            Object to up updated
        update_obj : dict|list[dict]|tuple[dict]
            When ``plotly_obj`` is an instance of :class:`BaseFigure`,
            ``update_obj`` should be a dict

            When ``plotly_obj`` is a tuple of instances of
            :class:`BasePlotlyType`, ``update_obj`` should be a tuple or list
            of dicts
        """
    from _plotly_utils.basevalidators import CompoundValidator, CompoundArrayValidator
    if update_obj is None:
        return
    elif isinstance(plotly_obj, BasePlotlyType):
        for key in update_obj:
            if key not in plotly_obj and isinstance(plotly_obj, BaseLayoutType):
                match = plotly_obj._subplot_re_match(key)
                if match:
                    plotly_obj[key] = {}
                    continue
            err = _check_path_in_prop_tree(plotly_obj, key, error_cast=ValueError)
            if err is not None:
                raise err
        if isinstance(update_obj, BasePlotlyType):
            update_obj = update_obj.to_plotly_json()
        for key in update_obj:
            val = update_obj[key]
            if overwrite:
                plotly_obj[key] = val
                continue
            validator = plotly_obj._get_prop_validator(key)
            if isinstance(validator, CompoundValidator) and isinstance(val, dict):
                BaseFigure._perform_update(plotly_obj[key], val)
            elif isinstance(validator, CompoundArrayValidator):
                if plotly_obj[key]:
                    BaseFigure._perform_update(plotly_obj[key], val)
                    if isinstance(val, (list, tuple)) and len(val) > len(plotly_obj[key]):
                        plotly_obj[key] = plotly_obj[key] + tuple(val[len(plotly_obj[key]):])
                else:
                    plotly_obj[key] = val
            else:
                plotly_obj[key] = val
    elif isinstance(plotly_obj, tuple):
        if len(update_obj) == 0:
            return
        else:
            for i, plotly_element in enumerate(plotly_obj):
                if isinstance(update_obj, dict):
                    if i in update_obj:
                        update_element = update_obj[i]
                    else:
                        continue
                else:
                    update_element = update_obj[i % len(update_obj)]
                BaseFigure._perform_update(plotly_element, update_element)
    else:
        raise ValueError('Unexpected plotly object with type {typ}'.format(typ=type(plotly_obj)))