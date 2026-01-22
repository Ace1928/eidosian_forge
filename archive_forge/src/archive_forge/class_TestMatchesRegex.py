import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestMatchesRegex(TestCase, TestMatchersInterface):
    matches_matcher = MatchesRegex('a|b')
    matches_matches = ['a', 'b']
    matches_mismatches = ['c']
    str_examples = [("MatchesRegex('a|b')", MatchesRegex('a|b')), ("MatchesRegex('a|b', re.M)", MatchesRegex('a|b', re.M)), ("MatchesRegex('a|b', re.I|re.M)", MatchesRegex('a|b', re.I | re.M)), ('MatchesRegex({!r})'.format(_b('§')), MatchesRegex(_b('§'))), ('MatchesRegex({!r})'.format('§'), MatchesRegex('§'))]
    describe_examples = [("'c' does not match /a|b/", 'c', MatchesRegex('a|b')), ("'c' does not match /a\\d/", 'c', MatchesRegex('a\\d')), ('{!r} does not match /\\s+\\xa7/'.format(_b('c')), _b('c'), MatchesRegex(_b('\\s+§'))), ('{!r} does not match /\\s+\\xa7/'.format('c'), 'c', MatchesRegex('\\s+§'))]