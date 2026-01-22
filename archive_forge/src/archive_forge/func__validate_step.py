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
def _validate_step(self, val, step):
    if step is not None:
        if not _is_number(step):
            raise ValueError(f'{_validate_error_prefix(self, 'step')} can only be None or a numeric value, not {type(step)}.')
        elif step == 0:
            raise ValueError(f'{_validate_error_prefix(self, 'step')} cannot be 0.')