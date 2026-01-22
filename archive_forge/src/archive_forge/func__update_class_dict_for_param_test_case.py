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
def _update_class_dict_for_param_test_case(test_class_name, dct, test_params_reprs, name, iterator):
    """Adds individual test cases to a dictionary.

  Args:
    test_class_name: The name of the class tests are added to.
    dct: The target dictionary.
    test_params_reprs: The dictionary for mapping names to test IDs.
    name: The original name of the test case.
    iterator: The iterator generating the individual test cases.

  Raises:
    DuplicateTestNameError: Raised when a test name occurs multiple times.
    RuntimeError: If non-parameterized functions are generated.
  """
    for idx, func in enumerate(iterator):
        assert callable(func), 'Test generators must yield callables, got %r' % (func,)
        if not (getattr(func, '__x_use_name__', None) or getattr(func, '__x_params_repr__', None)):
            raise RuntimeError('{}.{} generated a test function without using the parameterized decorators. Only tests generated using the decorators are supported.'.format(test_class_name, name))
        if getattr(func, '__x_use_name__', False):
            original_name = func.__name__
            new_name = original_name
        else:
            original_name = name
            new_name = '%s%d' % (original_name, idx)
        if new_name in dct:
            raise DuplicateTestNameError(test_class_name, new_name, original_name)
        dct[new_name] = func
        test_params_reprs[new_name] = getattr(func, '__x_params_repr__', '')