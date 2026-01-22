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
def _clean_repr(obj):
    return _ADDR_RE.sub('<\\1>', repr(obj))