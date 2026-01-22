import builtins
import dis
import opcode
import platform
import sys
import types
import weakref
import uuid
import threading
import typing
import warnings
from .compat import pickle
from collections import OrderedDict
from typing import ClassVar, Generic, Union, Tuple, Callable
from pickle import _getattribute
@instance
class _empty_cell_value:
    """sentinel for empty closures"""

    @classmethod
    def __reduce__(cls):
        return cls.__name__