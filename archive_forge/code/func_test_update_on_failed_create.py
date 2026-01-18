import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
def test_update_on_failed_create(self):
    broken_templ = self.main_template.replace('replace-this', 'true')
    stack_identifier = self.stack_create(template=broken_templ, files={'server_fail.yaml': self.nested_templ}, expected_status='CREATE_FAILED')
    fixed_templ = self.main_template.replace('replace-this', 'false')
    self.update_stack(stack_identifier, fixed_templ, files={'server_fail.yaml': self.nested_templ})