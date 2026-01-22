import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
class TestApplyScenario(testtools.TestCase):

    def setUp(self):
        super(TestApplyScenario, self).setUp()
        self.scenario_name = 'demo'
        self.scenario_attrs = {'foo': 'bar'}
        self.scenario = (self.scenario_name, self.scenario_attrs)

        class ReferenceTest(unittest.TestCase):

            def test_pass(self):
                pass

            def test_pass_with_docstring(self):
                """ The test that always passes.

                    This test case has a PEP 257 conformant docstring,
                    with its first line being a brief synopsis and the
                    rest of the docstring explaining that this test
                    does nothing but pass unconditionally.

                    """
                pass
        self.ReferenceTest = ReferenceTest

    def test_sets_specified_id(self):
        raw_test = self.ReferenceTest('test_pass')
        raw_id = 'ReferenceTest.test_pass'
        scenario_name = self.scenario_name
        expect_id = '%(raw_id)s(%(scenario_name)s)' % vars()
        modified_test = apply_scenario(self.scenario, raw_test)
        self.expectThat(modified_test.id(), EndsWith(expect_id))

    def test_sets_specified_attributes(self):
        raw_test = self.ReferenceTest('test_pass')
        modified_test = apply_scenario(self.scenario, raw_test)
        self.assertEqual('bar', modified_test.foo)

    def test_appends_scenario_name_to_short_description(self):
        raw_test = self.ReferenceTest('test_pass_with_docstring')
        modified_test = apply_scenario(self.scenario, raw_test)
        raw_doc = self.ReferenceTest.test_pass_with_docstring.__doc__
        raw_desc = raw_doc.split('\n')[0].strip()
        scenario_name = self.scenario_name
        expect_desc = '%(raw_desc)s (%(scenario_name)s)' % vars()
        self.assertEqual(expect_desc, modified_test.shortDescription())