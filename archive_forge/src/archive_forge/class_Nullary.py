import copy
import functools
import itertools
import sys
import types
import unittest
import warnings
from testtools.compat import reraise
from testtools import content
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.matchers._basic import _FlippedEquals
from testtools.monkey import patch
from testtools.runtest import (
from testtools.testresult import (
class Nullary:
    """Turn a callable into a nullary callable.

    The advantage of this over ``lambda: f(*args, **kwargs)`` is that it
    preserves the ``repr()`` of ``f``.
    """

    def __init__(self, callable_object, *args, **kwargs):
        self._callable_object = callable_object
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        return self._callable_object(*self._args, **self._kwargs)

    def __repr__(self):
        return repr(self._callable_object)