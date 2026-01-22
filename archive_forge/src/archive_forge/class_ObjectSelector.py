import collections
import copy
import datetime as dt
import glob
import inspect
import numbers
import os.path
import pathlib
import re
import sys
import typing
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from .parameterized import (
from ._utils import (
class ObjectSelector(Selector):
    """
    Deprecated. Same as Selector, but with a different constructor for
    historical reasons.
    """

    @typing.overload
    def __init__(self, default=None, *, objects=[], instantiate=False, compute_default_fn=None, check_on_set=None, allow_None=None, empty_default=False, doc=None, label=None, precedence=None, constant=False, readonly=False, pickle_default_value=True, per_instance=True, allow_refs=False, nested_refs=False):
        ...

    @_deprecate_positional_args
    def __init__(self, default=Undefined, *, objects=Undefined, **kwargs):
        super().__init__(objects=objects, default=default, empty_default=True, **kwargs)