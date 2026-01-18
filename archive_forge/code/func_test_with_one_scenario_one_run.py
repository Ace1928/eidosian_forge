import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
def test_with_one_scenario_one_run(self):

    class ReferenceTest(self.Implementation):
        scenarios = [('demo', {})]

        def test_pass(self):
            pass
    test = ReferenceTest('test_pass')
    log = []
    result = LoggingResult(log)
    test.run(result)
    self.assertTrue(result.wasSuccessful())
    self.assertEqual(1, result.testsRun)
    self.expectThat(log[0][1].id(), EndsWith('ReferenceTest.test_pass(demo)'))