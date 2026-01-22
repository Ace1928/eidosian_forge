import _collections_abc
import abc
import copyreg
import io
import itertools
import logging
import sys
import struct
import types
import weakref
import typing
from enum import Enum
from collections import ChainMap, OrderedDict
from .compat import pickle, Pickler
from .cloudpickle import (
def _cell_reduce(obj):
    """Cell (containing values of a function's free variables) reducer"""
    try:
        obj.cell_contents
    except ValueError:
        return (_make_empty_cell, ())
    else:
        return (_make_cell, (obj.cell_contents,))