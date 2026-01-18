import doctest
import io
import re
import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._datastructures import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_two_too_many_values(self):
    self.assertMismatchWithDescriptionMatching([1, 2, 3, 4], MatchesSetwise(Equals(1), Equals(2)), MatchesRegex('There were 2 values left over: \\[[34], [34]\\]'))