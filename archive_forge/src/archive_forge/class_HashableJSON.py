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
class HashableJSON(json.JSONEncoder):
    """
    Extends JSONEncoder to generate a hashable string for as many types
    of object as possible including nested objects and objects that are
    not normally hashable. The purpose of this class is to generate
    unique strings that once hashed are suitable for use in memoization
    and other cases where deep equality must be tested without storing
    the entire object.

    By default JSONEncoder supports booleans, numbers, strings, lists,
    tuples and dictionaries. In order to support other types such as
    sets, datetime objects and mutable objects such as pandas Dataframes
    or numpy arrays, HashableJSON has to convert these types to
    datastructures that can normally be represented as JSON.

    Support for other object types may need to be introduced in
    future. By default, unrecognized object types are represented by
    their id.

    One limitation of this approach is that dictionaries with composite
    keys (e.g. tuples) are not supported due to the JSON spec.
    """
    string_hashable = (dt.datetime,)
    repr_hashable = ()

    def default(self, obj):
        if isinstance(obj, set):
            return hash(frozenset(obj))
        elif isinstance(obj, np.ndarray):
            h = hashlib.new('md5')
            for s in obj.shape:
                h.update(_int_to_bytes(s))
            if obj.size >= _NP_SIZE_LARGE:
                state = np.random.RandomState(0)
                obj = state.choice(obj.flat, size=_NP_SAMPLE_SIZE)
            h.update(obj.tobytes())
            return h.hexdigest()
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            if len(obj) > _PANDAS_ROWS_LARGE:
                obj = obj.sample(n=_PANDAS_SAMPLE_SIZE, random_state=0)
            try:
                pd_values = list(pd.util.hash_pandas_object(obj, index=True).values)
            except TypeError:
                pd_values = [pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)]
            if isinstance(obj, pd.Series):
                columns = [obj.name]
            elif isinstance(obj.columns, pd.MultiIndex):
                columns = [name for cols in obj.columns for name in cols]
            else:
                columns = list(obj.columns)
            all_vals = pd_values + columns + list(obj.index.names)
            h = hashlib.md5()
            for val in all_vals:
                if not isinstance(val, bytes):
                    val = str(val).encode('utf-8')
                h.update(val)
            return h.hexdigest()
        elif isinstance(obj, self.string_hashable):
            return str(obj)
        elif isinstance(obj, self.repr_hashable):
            return repr(obj)
        try:
            return hash(obj)
        except Exception:
            return id(obj)