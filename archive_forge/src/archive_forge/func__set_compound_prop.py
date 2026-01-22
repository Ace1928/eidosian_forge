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
def _set_compound_prop(self, prop, val):
    """
        Set the value of a compound property

        Parameters
        ----------
        prop : str
            Name of a compound property
        val
            The new property value

        Returns
        -------
        BasePlotlyType
            The coerced assigned object
        """
    if val is Undefined:
        return
    validator = self._get_validator(prop)
    val = validator.validate_coerce(val, skip_invalid=self._skip_invalid)
    curr_val = self._compound_props.get(prop, None)
    if curr_val is not None:
        curr_dict_val = deepcopy(curr_val._props)
    else:
        curr_dict_val = None
    if val is not None:
        new_dict_val = deepcopy(val._props)
    else:
        new_dict_val = None
    if not self._in_batch_mode:
        if not new_dict_val:
            if self._props and prop in self._props:
                self._props.pop(prop)
        else:
            self._init_props()
            self._props[prop] = new_dict_val
    if not BasePlotlyType._vals_equal(curr_dict_val, new_dict_val):
        self._send_prop_set(prop, new_dict_val)
    if isinstance(val, BasePlotlyType):
        val._parent = self
        val._orphan_props.clear()
    if curr_val is not None:
        if curr_dict_val is not None:
            curr_val._orphan_props.update(curr_dict_val)
        curr_val._parent = None
    self._compound_props[prop] = val
    return val