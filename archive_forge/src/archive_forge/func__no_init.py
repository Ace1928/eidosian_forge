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
def _no_init(self, *args, **kwargs):
    if type(self)._is_protocol:
        raise TypeError('Protocols cannot be instantiated')