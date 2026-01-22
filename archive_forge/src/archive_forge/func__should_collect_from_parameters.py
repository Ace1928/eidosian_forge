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
def _should_collect_from_parameters(t):
    return isinstance(t, typing._GenericAlias) and (not t._special)