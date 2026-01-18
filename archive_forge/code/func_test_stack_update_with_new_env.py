import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_with_new_env(self):
    """Update handles new resource types in the environment.

        If a resource type appears during an update and the update fails,
        retrying the update is able to find the type properly in the
        environment.
        """
    stack_identifier = self.stack_create(template=test_template_one_resource)
    template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'fail': True})
    template['resources']['test2'] = {'type': 'My::TestResource'}
    template['resources']['test1']['depends_on'] = 'test2'
    env = {'resource_registry': {'My::TestResource': 'OS::Heat::TestResource'}}
    self.update_stack(stack_identifier, template=template, environment=env, expected_status='UPDATE_FAILED')
    template = _change_rsrc_properties(template, ['test1'], {'fail': False})
    template['resources']['test2']['properties'] = {'action_wait_secs': {'update': 1}}
    self.update_stack(stack_identifier, template=template, environment=env)
    self.assertEqual({'test1': 'OS::Heat::TestResource', 'test2': 'My::TestResource'}, self.list_resources(stack_identifier))