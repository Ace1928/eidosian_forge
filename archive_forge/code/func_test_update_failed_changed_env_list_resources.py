import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
@test.requires_convergence
def test_update_failed_changed_env_list_resources(self):
    template = {'heat_template_version': 'rocky', 'resources': {'test1': {'type': 'OS::Heat::TestResource', 'properties': {'value': 'foo'}}, 'my_res': {'type': 'My::TestResource', 'depends_on': 'test1'}, 'test2': {'depends_on': 'my_res', 'type': 'OS::Heat::TestResource'}}}
    env = {'resource_registry': {'My::TestResource': 'OS::Heat::TestResource'}}
    stack_identifier = self.stack_create(template=template, environment=env)
    update_template = copy.deepcopy(template)
    update_template['resources']['test1']['properties']['fail'] = 'true'
    update_template['resources']['test2']['depends_on'] = 'test1'
    del update_template['resources']['my_res']
    self.update_stack(stack_identifier, template=update_template, expected_status='UPDATE_FAILED')
    self.assertEqual(3, len(self.list_resources(stack_identifier)))