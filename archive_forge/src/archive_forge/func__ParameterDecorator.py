from the command line:
import functools
import re
import types
import unittest
import uuid
def _ParameterDecorator(naming_type, testcases):
    """Implementation of the parameterization decorators.

  Args:
    naming_type: The naming type.
    testcases: Testcase parameters.

  Returns:
    A function for modifying the decorated object.
  """

    def _Apply(obj):
        if isinstance(obj, type):
            _ModifyClass(obj, list(testcases) if not isinstance(testcases, collections_abc.Sequence) else testcases, naming_type)
            return obj
        else:
            return _ParameterizedTestIter(obj, testcases, naming_type)
    if _IsSingletonList(testcases):
        assert _NonStringIterable(testcases[0]), 'Single parameter argument must be a non-string iterable'
        testcases = testcases[0]
    return _Apply