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
def wrap_tuple_streams(unwrapped, kdims, streams):
    """
    Fills in tuple keys with dimensioned stream values as appropriate.
    """
    param_groups = [(s.contents.keys(), s) for s in streams]
    pairs = [(name, s) for group, s in param_groups for name in group]
    substituted = []
    for pos, el in enumerate(wrap_tuple(unwrapped)):
        if el is None and pos < len(kdims):
            matches = [(name, s) for name, s in pairs if name == kdims[pos].name]
            if len(matches) == 1:
                name, stream = matches[0]
                el = stream.contents[name]
        substituted.append(el)
    return tuple(substituted)