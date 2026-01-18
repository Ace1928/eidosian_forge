from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def test_patch_restored_after_run(self):
    self.foo = 'original'
    self.run_test(lambda case: case.patch(self, 'foo', 'patched'))
    self.assertThat(self.foo, Equals('original'))