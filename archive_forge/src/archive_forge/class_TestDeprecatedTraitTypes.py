import os
import sys
import tempfile
import textwrap
import shutil
import subprocess
import unittest
from traits.api import (
from traits.testing.optional_dependencies import requires_numpy
class TestDeprecatedTraitTypes(unittest.TestCase):

    def test_function_deprecated(self):

        def some_function():
            pass
        with self.assertWarnsRegex(DeprecationWarning, 'Function trait type'):
            Function()
        with self.assertWarnsRegex(DeprecationWarning, 'Function trait type'):
            Function(some_function, washable=True)

    def test_method_deprecated(self):

        class A:

            def some_method(self):
                pass
        with self.assertWarnsRegex(DeprecationWarning, 'Method trait type'):
            Method()
        with self.assertWarnsRegex(DeprecationWarning, 'Method trait type'):
            Method(A().some_method, gluten_free=False)

    def test_symbol_deprecated(self):
        with self.assertWarnsRegex(DeprecationWarning, 'Symbol trait type'):
            Symbol('random:random')