from the command line::
from collections import abc
import functools
import inspect
import itertools
import re
import types
import unittest
import warnings
from absl.testing import absltest
def _non_string_or_bytes_iterable(obj):
    return isinstance(obj, abc.Iterable) and (not isinstance(obj, str)) and (not isinstance(obj, bytes))