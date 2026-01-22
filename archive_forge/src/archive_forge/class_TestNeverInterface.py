from testtools import TestCase
from testtools.matchers import Always, Never
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestNeverInterface(TestMatchersInterface, TestCase):
    """:py:func:`~testtools.matchers.Never` never matches."""
    matches_matcher = Never()
    matches_matches = []
    matches_mismatches = [42, object(), 'hi mom']
    str_examples = [('Never()', Never())]
    describe_examples = [('Inevitable mismatch on 42', 42, Never())]