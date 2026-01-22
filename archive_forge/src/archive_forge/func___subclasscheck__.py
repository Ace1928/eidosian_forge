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
def __subclasscheck__(cls, other):
    raise TypeError('TypedDict does not support instance and class checks')