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
def _length_bounds_check(self, bounds, length, name):
    message = f'{name} length {length} does not match declared bounds of {bounds}'
    if not isinstance(bounds, tuple):
        if bounds != length:
            raise ValueError(f'{_validate_error_prefix(self)}: {message}')
        else:
            return
    lower, upper = bounds
    failure = lower is not None and length < lower or (upper is not None and length > upper)
    if failure:
        raise ValueError(f'{_validate_error_prefix(self)}: {message}')