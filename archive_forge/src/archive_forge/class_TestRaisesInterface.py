import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._exception import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestRaisesInterface(TestCase, TestMatchersInterface):
    matches_matcher = Raises()

    def boom():
        raise Exception('foo')
    matches_matches = [boom]
    matches_mismatches = [lambda: None]
    str_examples = []
    describe_examples = []