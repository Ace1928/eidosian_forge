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
def _validate_item_type(self, val, item_type):
    if item_type is None or (self.allow_None and val is None):
        return
    for v in val:
        if isinstance(v, item_type):
            continue
        raise TypeError(f'{_validate_error_prefix(self)} items must be instances of {item_type!r}, not {type(v)}.')