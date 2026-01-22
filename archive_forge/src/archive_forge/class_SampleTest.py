import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
class SampleTest(unittest.TestCase):

    def test_nothing(self):
        pass
    scenarios = [('a', {}), ('b', {})]