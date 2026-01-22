from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._higherorder import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestAnnotatedMismatch(TestCase):
    run_tests_with = FullStackRunTest

    def test_forwards_details(self):
        x = Mismatch('description', {'foo': 'bar'})
        annotated = AnnotatedMismatch('annotation', x)
        self.assertEqual(x.get_details(), annotated.get_details())