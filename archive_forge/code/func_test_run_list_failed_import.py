import doctest
import io
import sys
from textwrap import dedent
import unittest
from unittest import TestSuite
import testtools
from testtools import TestCase, run, skipUnless
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools import TestCase
from fixtures import Fixture
from testresources import (
from testtools import TestCase
from testtools import TestCase, clone_test_with_new_id
def test_run_list_failed_import(self):
    broken = self.useFixture(SampleTestFixture(broken=True))
    out = io.StringIO()
    unittest.defaultTestLoader._top_level_dir = None
    exc = self.assertRaises(SystemExit, run.main, ['prog', 'discover', '-l', broken.package.base, '*.py'], out)
    self.assertEqual(2, exc.args[0])
    self.assertThat(out.getvalue(), DocTestMatches('unittest.loader._FailedTest.runexample\nFailed to import test module: runexample\nTraceback (most recent call last):\n  File ".../loader.py", line ..., in _find_test_path\n    package = self._get_module_from_name(name)...\n  File ".../loader.py", line ..., in _get_module_from_name\n    __import__(name)...\n  File ".../runexample/__init__.py", line 1\n    class not in\n...^...\nSyntaxError: invalid syntax\n\n', doctest.ELLIPSIS))