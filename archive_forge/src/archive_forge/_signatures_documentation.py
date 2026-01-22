import functools
import inspect
import itertools
import operator
from importlib import import_module
from .functoolz import (is_partial_args, is_arity, has_varargs,
import builtins
 Like ``is_partial_args`` for builtins in our ``signatures`` registry