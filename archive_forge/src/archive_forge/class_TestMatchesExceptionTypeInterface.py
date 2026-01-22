import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._exception import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestMatchesExceptionTypeInterface(TestCase, TestMatchersInterface):
    matches_matcher = MatchesException(ValueError)
    error_foo = make_error(ValueError, 'foo')
    error_sub = make_error(UnicodeError, 'bar')
    error_base_foo = make_error(Exception, 'foo')
    matches_matches = [error_foo, error_sub]
    matches_mismatches = [error_base_foo]
    str_examples = [('MatchesException(%r)' % Exception, MatchesException(Exception))]
    describe_examples = [(f'{Exception!r} is not a {ValueError!r}', error_base_foo, MatchesException(ValueError))]