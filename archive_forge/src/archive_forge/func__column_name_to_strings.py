import ast
from collections.abc import Sequence
from concurrent import futures
import concurrent.futures.thread  # noqa
from copy import deepcopy
from itertools import zip_longest
import json
import operator
import re
import warnings
import numpy as np
import pyarrow as pa
from pyarrow.lib import _pandas_api, frombytes  # noqa
def _column_name_to_strings(name):
    """Convert a column name (or level) to either a string or a recursive
    collection of strings.

    Parameters
    ----------
    name : str or tuple

    Returns
    -------
    value : str or tuple

    Examples
    --------
    >>> name = 'foo'
    >>> _column_name_to_strings(name)
    'foo'
    >>> name = ('foo', 'bar')
    >>> _column_name_to_strings(name)
    "('foo', 'bar')"
    >>> import pandas as pd
    >>> name = (1, pd.Timestamp('2017-02-01 00:00:00'))
    >>> _column_name_to_strings(name)
    "('1', '2017-02-01 00:00:00')"
    """
    if isinstance(name, str):
        return name
    elif isinstance(name, bytes):
        return name.decode('utf8')
    elif isinstance(name, tuple):
        return str(tuple(map(_column_name_to_strings, name)))
    elif isinstance(name, Sequence):
        raise TypeError('Unsupported type for MultiIndex level')
    elif name is None:
        return None
    return str(name)