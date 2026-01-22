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
def _str_to_dict_path_full(key_path_str):
    """
    Convert a key path string into a tuple of key path elements and also
    return a tuple of indices marking the beginning of each element in the
    string.

    Parameters
    ----------
    key_path_str : str
        Key path string, where nested keys are joined on '.' characters
        and array indexes are specified using brackets
        (e.g. 'foo.bar[1]')
    Returns
    -------
    tuple[str | int]
    tuple [int]
    """
    if len(key_path_str):
        key_path2 = split_multichar([key_path_str], list('.[]'))
        key_path3 = []
        underscore_props = BaseFigure._valid_underscore_properties

        def _make_hyphen_key(key):
            if '_' in key[1:]:
                for under_prop, hyphen_prop in underscore_props.items():
                    key = key.replace(under_prop, hyphen_prop)
            return key

        def _make_underscore_key(key):
            return key.replace('-', '_')
        key_path2b = list(map(_make_hyphen_key, key_path2))

        def _split_and_chomp(s):
            if not len(s):
                return s
            s_split = split_multichar([s], list('_'))
            s_chomped = chomp_empty_strings(s_split, '_', reverse=True)
            return s_chomped
        key_path2c = list(reduce(lambda x, y: x + y if type(y) == type(list()) else x + [y], map(_split_and_chomp, key_path2b), []))
        key_path2d = list(map(_make_underscore_key, key_path2c))
        all_elem_idcs = tuple(split_string_positions(list(key_path2d)))
        key_elem_pairs = list(filter(lambda t: len(t[1]), enumerate(key_path2d)))
        key_path3 = [x for _, x in key_elem_pairs]
        elem_idcs = [all_elem_idcs[i] for i, _ in key_elem_pairs]
        for i in range(len(key_path3)):
            try:
                key_path3[i] = int(key_path3[i])
            except ValueError as _:
                pass
    else:
        key_path3 = []
        elem_idcs = []
    return (tuple(key_path3), elem_idcs)