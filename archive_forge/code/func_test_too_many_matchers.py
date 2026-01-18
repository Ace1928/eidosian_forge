import doctest
import io
import re
import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._datastructures import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_too_many_matchers(self):
    self.assertMismatchWithDescriptionMatching([2, 3], MatchesSetwise(Equals(1), Equals(2), Equals(3)), Equals('There was 1 matcher left over: Equals(1)'))