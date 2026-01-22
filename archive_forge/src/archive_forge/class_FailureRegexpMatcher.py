import collections
import logging
from unittest import mock
import fixtures
from oslotest import base
from testtools import compat
from testtools import matchers
from testtools import testcase
from taskflow import exceptions
from taskflow.tests import fixtures as taskflow_fixtures
from taskflow.tests import utils
from taskflow.utils import misc
class FailureRegexpMatcher(object):
    """Matches if the failure was caused by the given exception and message.

    This will match if a given failure contains and exception of the given
    class type and if its string message matches to the given regular
    expression pattern.
    """

    def __init__(self, exc_class, pattern):
        self.exc_class = exc_class
        self.pattern = pattern

    def match(self, failure):
        for cause in failure:
            if cause.check(self.exc_class) is not None:
                return matchers.MatchesRegex(self.pattern).match(cause.exception_str)
        return matchers.Mismatch("The `%s` wasn't caused by the `%s`" % (failure, self.exc_class))