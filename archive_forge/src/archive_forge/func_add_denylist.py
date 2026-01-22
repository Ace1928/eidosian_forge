import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def add_denylist(mod, attr):
    if hasattr(mod, attr):
        denylist.add(getattr(mod, attr))