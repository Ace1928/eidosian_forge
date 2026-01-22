import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestIsInterface(TestCase, TestMatchersInterface):
    foo = object()
    bar = object()
    matches_matcher = Is(foo)
    matches_matches = [foo]
    matches_mismatches = [bar, 1]
    str_examples = [('Is(2)', Is(2))]
    describe_examples = [('2 is not 1', 2, Is(1))]