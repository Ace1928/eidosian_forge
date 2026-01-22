import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
class TemplateResourceErrorMessageTest(functional_base.FunctionalTestsBase):
    """Prove that nested stack errors don't suck."""
    template = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  victim:\n    Type: fail.yaml\n"
    nested_templ = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  oops:\n    Type: OS::Heat::TestResource\n    Properties:\n      fail: true\n      wait_secs: 2\n"

    def test_fail(self):
        stack_identifier = self.stack_create(template=self.template, files={'fail.yaml': self.nested_templ}, expected_status='CREATE_FAILED')
        stack = self.client.stacks.get(stack_identifier)
        exp_path = 'resources.victim.resources.oops'
        exp_msg = 'Test Resource failed oops'
        exp = 'Resource CREATE failed: ValueError: %s: %s' % (exp_path, exp_msg)
        self.assertEqual(exp, stack.stack_status_reason)