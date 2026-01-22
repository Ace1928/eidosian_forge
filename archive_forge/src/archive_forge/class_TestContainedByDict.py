from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._dict import (
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestContainedByDict(TestCase, TestMatchersInterface):
    matches_matcher = ContainedByDict({'foo': Equals('bar'), 'baz': Not(Equals('qux'))})
    matches_matches = [{}, {'foo': 'bar'}, {'foo': 'bar', 'baz': 'quux'}, {'baz': 'quux'}]
    matches_mismatches = [{'foo': 'bar', 'baz': 'quux', 'cat': 'dog'}, {'foo': 'bar', 'baz': 'qux'}, {'foo': 'bop', 'baz': 'qux'}, {'foo': 'bar', 'cat': 'dog'}]
    str_examples = [("ContainedByDict({{'baz': {}, 'foo': {}}})".format(Not(Equals('qux')), Equals('bar')), matches_matcher)]
    describe_examples = [("Differences: {\n  'baz': 'qux' matches Equals('qux'),\n}", {'foo': 'bar', 'baz': 'qux'}, matches_matcher), ("Differences: {\n  'baz': 'qux' matches Equals('qux'),\n  'foo': 'bop' != 'bar',\n}", {'foo': 'bop', 'baz': 'qux'}, matches_matcher), ("Extra: {\n  'cat': 'dog',\n}", {'foo': 'bar', 'cat': 'dog'}, matches_matcher)]