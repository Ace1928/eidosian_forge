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
def _is_unpack(obj):
    return isinstance(obj, _UnpackAlias)