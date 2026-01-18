import doctest
import io
import re
import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._datastructures import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_two_too_many_matchers(self):
    self.assertMismatchWithDescriptionMatching([3], MatchesSetwise(Equals(1), Equals(2), Equals(3)), MatchesRegex('There were 2 matchers left over: Equals\\([12]\\), Equals\\([12]\\)'))