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
def _validate_order(self, val, step, allow_None):
    if val is None and allow_None:
        return
    elif val is not None and (val[0] is None or val[1] is None):
        return
    start, end = val
    if step is not None and step > 0 and (not start <= end):
        raise ValueError(f'{_validate_error_prefix(self)} end {end} is less than its start {start} with positive step {step}.')
    elif step is not None and step < 0 and (not start >= end):
        raise ValueError(f'{_validate_error_prefix(self)} start {start} is less than its start {end} with negative step {step}.')