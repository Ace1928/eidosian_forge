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
def _validate_regex(self, val, regex):
    if val is None and self.allow_None:
        return
    if regex is not None and re.match(regex, val) is None:
        raise ValueError(f'{_validate_error_prefix(self)} value {val!r} does not match regex {regex!r}.')