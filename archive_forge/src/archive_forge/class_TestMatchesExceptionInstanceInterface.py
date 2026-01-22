import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._exception import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestMatchesExceptionInstanceInterface(TestCase, TestMatchersInterface):
    matches_matcher = MatchesException(ValueError('foo'))
    error_foo = make_error(ValueError, 'foo')
    error_bar = make_error(ValueError, 'bar')
    error_base_foo = make_error(Exception, 'foo')
    matches_matches = [error_foo]
    matches_mismatches = [error_bar, error_base_foo]
    if sys.version_info >= (3, 7):
        _e = ''
    else:
        _e = ','
    str_examples = [("MatchesException(Exception('foo'%s))" % _e, MatchesException(Exception('foo')))]
    describe_examples = [(f'{Exception!r} is not a {ValueError!r}', error_base_foo, MatchesException(ValueError('foo'))), ("ValueError('bar'%s) has different arguments to ValueError('foo'%s)." % (_e, _e), error_bar, MatchesException(ValueError('foo')))]