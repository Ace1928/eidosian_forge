import itertools
import types
import unittest
from cupy.testing import _bundle
from cupy.testing import _pytest_impl
Generates test classes with given sets of additional attributes

    >>> @parameterize({"a": 1}, {"b": 2, "c": 3})
    ... class TestX(unittest.TestCase):
    ...     def test_y(self):
    ...         pass

    generates two classes `TestX_param_0_...`, `TestX_param_1_...` and
    removes the original class `TestX`.

    The specification is subject to change, which applies to all the non-NumPy
    `testing` features.

    