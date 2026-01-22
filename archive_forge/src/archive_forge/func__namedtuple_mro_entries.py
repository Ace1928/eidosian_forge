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
def _namedtuple_mro_entries(bases):
    assert NamedTuple in bases
    return (_NamedTuple,)