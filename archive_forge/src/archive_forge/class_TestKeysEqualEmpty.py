from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._dict import (
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestKeysEqualEmpty(TestCase, TestMatchersInterface):
    matches_matcher = KeysEqual()
    matches_matches = [{}]
    matches_mismatches = [{'foo': 0, 'bar': 1}, {'foo': 0}, {'bar': 1}, {'foo': 0, 'bar': 1, 'baz': 2}, {'a': None, 'b': None, 'c': None}]
    str_examples = [('KeysEqual()', KeysEqual())]
    describe_examples = [('[] does not match {1: 2}: Keys not equal', {1: 2}, matches_matcher)]