import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
def test_tests_with_scenarios_calls_apply_scenarios(self):

    class ReferenceTest(unittest.TestCase):
        scenarios = [('demo', {})]

        def test_pass(self):
            pass
    test = ReferenceTest('test_pass')
    log = self.hook_apply_scenarios()
    tests = list(generate_scenarios(test))
    self.expectThat(tests[0].id(), EndsWith('ReferenceTest.test_pass(demo)'))
    self.assertEqual([([('demo', {})], test)], log)