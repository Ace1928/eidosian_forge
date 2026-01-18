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
def test_successive_patches_restored_after_run(self):
    self.foo = 'original'

    def test_body(case):
        case.patch(self, 'foo', 'patched')
        case.patch(self, 'foo', 'second')
        return self.foo
    self.run_test(test_body)
    self.assertThat(self.foo, Equals('original'))