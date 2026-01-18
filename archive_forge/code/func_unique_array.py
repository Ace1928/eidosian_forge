import builtins
import datetime as dt
import hashlib
import inspect
import itertools
import json
import numbers
import operator
import pickle
import string
import sys
import time
import types
import unicodedata
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import partial
from threading import Event, Thread
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
def unique_array(arr):
    """
    Returns an array of unique values in the input order.

    Args:
       arr (np.ndarray or list): The array to compute unique values on

    Returns:
       A new array of unique values
    """
    if not len(arr):
        return np.asarray(arr)
    if isinstance(arr, np.ndarray) and arr.dtype.kind not in 'MO':
        return pd.unique(arr)
    values = []
    for v in arr:
        if isinstance(v, datetime_types) and (not isinstance(v, cftime_types)):
            v = pd.Timestamp(v).to_datetime64()
        elif isinstance(getattr(v, 'dtype', None), pd.CategoricalDtype):
            v = v.dtype.categories
        values.append(v)
    return pd.unique(np.asarray(values).ravel())