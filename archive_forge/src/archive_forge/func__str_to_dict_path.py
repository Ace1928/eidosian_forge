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
def _str_to_dict_path(key_path_str):
    """
        Convert a key path string into a tuple of key path elements.

        Parameters
        ----------
        key_path_str : str
            Key path string, where nested keys are joined on '.' characters
            and array indexes are specified using brackets
            (e.g. 'foo.bar[1]')
        Returns
        -------
        tuple[str | int]
        """
    if isinstance(key_path_str, str) and '.' not in key_path_str and ('[' not in key_path_str) and ('_' not in key_path_str):
        return (key_path_str,)
    elif isinstance(key_path_str, tuple):
        return key_path_str
    else:
        ret = _str_to_dict_path_full(key_path_str)[0]
        return ret