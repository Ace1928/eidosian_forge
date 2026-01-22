import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestLessThanInterface(TestCase, TestMatchersInterface):
    matches_matcher = LessThan(4)
    matches_matches = [-5, 3]
    matches_mismatches = [4, 5, 5000]
    str_examples = [('LessThan(12)', LessThan(12))]
    describe_examples = [('5 >= 4', 5, LessThan(4)), ('4 >= 4', 4, LessThan(4))]