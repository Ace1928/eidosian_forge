import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
def _value_and_type_iter(params):
    for p in params:
        yield (p, type(p))