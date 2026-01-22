from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._dict import (
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestKeysEqualWithList(TestCase, TestMatchersInterface):
    matches_matcher = KeysEqual('foo', 'bar')
    matches_matches = [{'foo': 0, 'bar': 1}]
    matches_mismatches = [{}, {'foo': 0}, {'bar': 1}, {'foo': 0, 'bar': 1, 'baz': 2}, {'a': None, 'b': None, 'c': None}]
    str_examples = [("KeysEqual('foo', 'bar')", KeysEqual('foo', 'bar'))]
    describe_examples = []

    def test_description(self):
        matchee = {'foo': 0, 'bar': 1, 'baz': 2}
        mismatch = KeysEqual('foo', 'bar').match(matchee)
        description = mismatch.describe()
        self.assertThat(description, Equals("['bar', 'foo'] does not match %r: Keys not equal" % (matchee,)))