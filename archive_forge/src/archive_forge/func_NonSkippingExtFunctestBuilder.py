from __future__ import print_function
import sys
import types
import unittest
import google3
from absl import flags
from dulwich import tests
from dulwich.tests import utils
from google3.devtools.git.common import (  # pylint: disable-msg=C6204
from google3.testing.pybase import googletest  # pylint: disable-msg=W0611
def NonSkippingExtFunctestBuilder(method, func):
    """Alternate implementation of dulwich.tests.utils.ext_functest_builder.

  Dulwich skips extension tests for missing C extensions, but we need them in
  google3. This implementation fails fast if the C extensions are not found.

  Args:
    method: The method to run.
    func: The function implementation to pass to method.

  Returns:
    A test method to run the given C extension function.
  """

    def DoTest(self):
        self.assertTrue(isinstance(func, types.BuiltinFunctionType), 'C extension for %s not found' % func.__name__)
        method(self, func)
    return DoTest