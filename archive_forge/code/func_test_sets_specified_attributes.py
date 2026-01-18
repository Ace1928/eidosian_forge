import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
def test_sets_specified_attributes(self):
    raw_test = self.ReferenceTest('test_pass')
    modified_test = apply_scenario(self.scenario, raw_test)
    self.assertEqual('bar', modified_test.foo)