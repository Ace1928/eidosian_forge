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
def _parameter_decorator(naming_type, testcases):
    """Implementation of the parameterization decorators.

  Args:
    naming_type: The naming type.
    testcases: Testcase parameters.

  Raises:
    NoTestsError: Raised when the decorator generates no tests.

  Returns:
    A function for modifying the decorated object.
  """

    def _apply(obj):
        if isinstance(obj, type):
            _modify_class(obj, testcases, naming_type)
            return obj
        else:
            return _ParameterizedTestIter(obj, testcases, naming_type)
    if len(testcases) == 1 and (not isinstance(testcases[0], tuple)) and (not isinstance(testcases[0], abc.Mapping)):
        assert _non_string_or_bytes_iterable(testcases[0]), 'Single parameter argument must be a non-string non-Mapping iterable'
        testcases = testcases[0]
    if not isinstance(testcases, abc.Sequence):
        testcases = list(testcases)
    if not testcases:
        raise NoTestsError('parameterized test decorators did not generate any tests. Make sure you specify non-empty parameters, and do not reuse generators more than once.')
    return _apply