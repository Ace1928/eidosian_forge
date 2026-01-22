import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
class TestMultiplyScenarios(testtools.TestCase):

    def test_multiply_scenarios(self):

        def factory(name):
            for i in 'ab':
                yield (i, {name: i})
        scenarios = multiply_scenarios(factory('p'), factory('q'))
        self.assertEqual([('a,a', dict(p='a', q='a')), ('a,b', dict(p='a', q='b')), ('b,a', dict(p='b', q='a')), ('b,b', dict(p='b', q='b'))], scenarios)

    def test_multiply_many_scenarios(self):

        def factory(name):
            for i in 'abc':
                yield (i, {name: i})
        scenarios = multiply_scenarios(factory('p'), factory('q'), factory('r'), factory('t'))
        self.assertEqual(3 ** 4, len(scenarios), scenarios)
        self.assertEqual('a,a,a,a', scenarios[0][0])