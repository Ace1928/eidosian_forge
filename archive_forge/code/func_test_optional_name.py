import io
import os
import tempfile
import unittest
from testtools import TestCase
from testtools.compat import (
from testtools.content import (
from testtools.content_type import (
from testtools.matchers import (
from testtools.tests.helpers import an_exc_info
def test_optional_name(self):

    class SomeTest(TestCase):

        def test_foo(self):
            pass
    test = SomeTest('test_foo')
    path = self.make_file('some data')
    base_path = os.path.basename(path)
    attach_file(test, path)
    self.assertEqual([base_path], list(test.getDetails()))