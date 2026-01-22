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
def CoopTestCase(other_base_class):
    """Returns a new base class with a cooperative metaclass base.

  This enables the TestCase to be used in combination
  with other base classes that have custom metaclasses, such as
  ``mox.MoxTestBase``.

  Only works with metaclasses that do not override ``type.__new__``.

  Example::

      from absl.testing import parameterized

      class ExampleTest(parameterized.CoopTestCase(OtherTestCase)):
        ...

  Args:
    other_base_class: (class) A test case base class.

  Returns:
    A new class object.
  """
    if type(other_base_class) == type:
        warnings.warn(f'CoopTestCase is only necessary when combining with a class that uses a metaclass. Use multiple inheritance like this instead: class ExampleTest(paramaterized.TestCase, {other_base_class.__name__}):', stacklevel=2)

        class CoopTestCaseBase(other_base_class, TestCase):
            pass
        return CoopTestCaseBase
    else:

        class CoopMetaclass(type(other_base_class), TestGeneratorMetaclass):
            pass

        class CoopTestCaseBase(other_base_class, TestCase, metaclass=CoopMetaclass):
            pass
        return CoopTestCaseBase