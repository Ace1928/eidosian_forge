import inspect
import itertools
import os
import warnings
from functools import partial
from importlib.metadata import entry_points
import networkx as nx
from .decorators import argmap
from .configs import Config, config
def _dispatchable(func=None, **kwargs):
    if func is None:
        return partial(_dispatchable, **kwargs)
    dispatched_func = _orig_dispatchable(func, **kwargs)
    func.__doc__ = dispatched_func.__doc__
    return func