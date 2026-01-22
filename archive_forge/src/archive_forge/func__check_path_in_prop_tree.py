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
def _check_path_in_prop_tree(obj, path, error_cast=None):
    """
    obj:        the object in which the first property is looked up
    path:       the path that will be split into properties to be looked up
                path can also be a tuple. In this case, it is combined using .
                and [] because it is impossible to reconstruct the string fully
                in order to give a decent error message.
    error_cast: this function walks down the property tree by looking up values
                in objects. So this will throw exceptions that are thrown by
                __getitem__, but in some cases we are checking the path for a
                different reason and would prefer throwing a more relevant
                exception (e.g., __getitem__ throws KeyError but __setitem__
                throws ValueError for subclasses of BasePlotlyType and
                BaseFigure). So the resulting error can be "casted" to the
                passed in type, if not None.
    returns
          an Exception object or None. The caller can raise this
          exception to see where the lookup error occurred.
    """
    if isinstance(path, tuple):
        path = _remake_path_from_tuple(path)
    prop, prop_idcs = _str_to_dict_path_full(path)
    prev_objs = []
    for i, p in enumerate(prop):
        arg = ''
        prev_objs.append(obj)
        try:
            obj = obj[p]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            arg = e.args[0]
            if issubclass(e.__class__, TypeError):
                if i > 0:
                    validator = prev_objs[i - 1]._get_validator(prop[i - 1])
                    arg += "\n\nInvalid value received for the '{plotly_name}' property of {parent_name}\n\n{description}".format(parent_name=validator.parent_name, plotly_name=validator.plotly_name, description=validator.description())
                disp_i = max(i - 1, 0)
                dict_item_len = _len_dict_item(prop[disp_i])
                trailing_underscores = ''
                if prop[i][0] == '_':
                    trailing_underscores = ' and path has trailing underscores'
                if trailing_underscores != '' and disp_i != i:
                    dict_item_len += _len_dict_item(prop[i])
                arg += '\n\nProperty does not support subscripting%s:\n%s\n%s' % (trailing_underscores, path, display_string_positions(prop_idcs, disp_i, length=dict_item_len, char='^'))
            else:
                arg += '\nBad property path:\n%s\n%s' % (path, display_string_positions(prop_idcs, i, length=_len_dict_item(prop[i]), char='^'))
            if isinstance(e, KeyError):
                e = PlotlyKeyError()
            if error_cast is not None:
                e = error_cast()
            e.args = (arg,)
            return e
    return None