import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
def test_per_module_scenarios(self):
    """Generate scenarios for available modules"""
    s = testscenarios.scenarios.per_module_scenarios('the_module', [('Python', 'testscenarios'), ('unittest', 'unittest'), ('nonexistent', 'nonexistent')])
    self.assertEqual('nonexistent', s[-1][0])
    self.assertIsInstance(s[-1][1]['the_module'], tuple)
    s[-1][1]['the_module'] = None
    self.assertEqual(s, [('Python', {'the_module': testscenarios}), ('unittest', {'the_module': unittest}), ('nonexistent', {'the_module': None})])