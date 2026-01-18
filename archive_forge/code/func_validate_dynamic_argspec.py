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
def validate_dynamic_argspec(callback, kdims, streams):
    """
    Utility used by DynamicMap to ensure the supplied callback has an
    appropriate signature.

    If validation succeeds, returns a list of strings to be zipped with
    the positional arguments, i.e. kdim values. The zipped values can then
    be merged with the stream values to pass everything to the Callable
    as keywords.

    If the callbacks use *args, None is returned to indicate that kdim
    values must be passed to the Callable by position. In this
    situation, Callable passes *args and **kwargs directly to the
    callback.

    If the callback doesn't use **kwargs, the accepted keywords are
    validated against the stream parameter names.
    """
    argspec = callback.argspec
    name = callback.name
    kdims = [kdim.name for kdim in kdims]
    stream_params = stream_parameters(streams)
    defaults = argspec.defaults if argspec.defaults else []
    all_posargs = argspec.args[:-len(defaults)] if defaults else argspec.args
    posargs = [arg for arg in all_posargs if arg not in stream_params]
    kwargs = argspec.args[-len(defaults):]
    if argspec.keywords is None:
        unassigned_streams = set(stream_params) - set(argspec.args)
        if unassigned_streams:
            unassigned = ','.join(unassigned_streams)
            raise KeyError(f'Callable {name!r} missing keywords to accept stream parameters: {unassigned}')
    if len(posargs) > len(kdims) + len(stream_params):
        raise KeyError(f'Callable {name!r} accepts more positional arguments than there are kdims and stream parameters')
    if kdims == []:
        return []
    if set(kdims) == set(posargs):
        return kdims
    elif len(posargs) == len(kdims):
        if argspec.args[:len(kdims)] != posargs:
            raise KeyError(f'Unmatched positional kdim arguments only allowed at the start of the signature of {name!r}')
        return posargs
    elif argspec.varargs:
        return None
    elif set(posargs) - set(kdims):
        raise KeyError(f'Callable {name!r} accepts more positional arguments {posargs} than there are key dimensions {kdims}')
    elif set(kdims).issubset(set(kwargs)):
        return kdims
    elif set(kdims).issubset(set(posargs + kwargs)):
        return kdims
    elif argspec.keywords:
        return kdims
    else:
        names = list(set(posargs + kwargs))
        raise KeyError(f'Callback {name!r} signature over {names} does not accommodate required kdims {kdims}')