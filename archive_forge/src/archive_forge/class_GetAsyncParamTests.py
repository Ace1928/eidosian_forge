import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
class GetAsyncParamTests(SynchronousTestCase):
    """
    Tests for L{twisted.python.compat._get_async_param}
    """

    def test_get_async_param(self):
        """
        L{twisted.python.compat._get_async_param} uses isAsync by default,
        or deprecated async keyword argument if isAsync is None.
        """
        self.assertEqual(_get_async_param(isAsync=False), False)
        self.assertEqual(_get_async_param(isAsync=True), True)
        self.assertEqual(_get_async_param(isAsync=None, **{'async': False}), False)
        self.assertEqual(_get_async_param(isAsync=None, **{'async': True}), True)
        self.assertRaises(TypeError, _get_async_param, False, {'async': False})

    def test_get_async_param_deprecation(self):
        """
        L{twisted.python.compat._get_async_param} raises a deprecation
        warning if async keyword argument is passed.
        """
        self.assertEqual(_get_async_param(isAsync=None, **{'async': False}), False)
        currentWarnings = self.flushWarnings(offendingFunctions=[self.test_get_async_param_deprecation])
        self.assertEqual(currentWarnings[0]['message'], "'async' keyword argument is deprecated, please use isAsync")