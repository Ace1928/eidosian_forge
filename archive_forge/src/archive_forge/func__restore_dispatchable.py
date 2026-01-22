import inspect
import itertools
import os
import warnings
from functools import partial
from importlib.metadata import entry_points
import networkx as nx
from .decorators import argmap
from .configs import Config, config
def _restore_dispatchable(name):
    return _registered_algorithms[name]