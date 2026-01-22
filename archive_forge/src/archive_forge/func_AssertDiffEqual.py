import os
import difflib
import unittest
import six
from apitools.gen import gen_client
from apitools.gen import test_utils
def AssertDiffEqual(self, expected, actual):
    """Like unittest.assertEqual with a diff in the exception message."""
    if expected != actual:
        unified_diff = difflib.unified_diff(expected.splitlines(), actual.splitlines())
        raise AssertionError('\n'.join(unified_diff))