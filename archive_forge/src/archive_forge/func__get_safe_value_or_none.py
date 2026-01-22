import copy
import sys
import re
import os
from itertools import chain
from contextlib import contextmanager
from parso.python import tree
def _get_safe_value_or_none(value, accept):
    value = value.get_safe_value(default=None)
    if isinstance(value, accept):
        return value