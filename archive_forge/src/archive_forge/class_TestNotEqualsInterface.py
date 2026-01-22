import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestNotEqualsInterface(TestCase, TestMatchersInterface):
    matches_matcher = NotEquals(1)
    matches_matches = [2]
    matches_mismatches = [1]
    str_examples = [('NotEquals(1)', NotEquals(1)), ("NotEquals('1')", NotEquals('1'))]
    describe_examples = [('1 == 1', 1, NotEquals(1))]