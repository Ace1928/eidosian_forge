import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
class Reference2(unittest.TestCase):
    scenarios = [('3', {}), ('4', {})]

    def test_something(self):
        pass