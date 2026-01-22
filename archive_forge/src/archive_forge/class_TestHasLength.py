import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestHasLength(TestCase, TestMatchersInterface):
    matches_matcher = HasLength(2)
    matches_matches = [[1, 2]]
    matches_mismatches = [[], [1], [3, 2, 1]]
    str_examples = [('HasLength(2)', HasLength(2))]
    describe_examples = [('len([]) != 1', [], HasLength(1))]