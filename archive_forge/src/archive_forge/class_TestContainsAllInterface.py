import doctest
import io
import re
import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._datastructures import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestContainsAllInterface(TestCase, TestMatchersInterface):
    matches_matcher = ContainsAll(['foo', 'bar'])
    matches_matches = [['foo', 'bar'], ['foo', 'z', 'bar'], ['bar', 'foo']]
    matches_mismatches = [['f', 'g'], ['foo', 'baz'], []]
    str_examples = [("MatchesAll(Contains('foo'), Contains('bar'))", ContainsAll(['foo', 'bar']))]
    describe_examples = [("Differences: [\n'baz' not in 'foo'\n]", 'foo', ContainsAll(['foo', 'baz']))]