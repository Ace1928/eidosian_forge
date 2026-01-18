from heatclient import exc
import keystoneclient
from heat_integrationtests.functional import functional_base
def test_unavailable_resources_not_created(self):
    stack_name = self._stack_rand_name()
    parameters = {'instance_type': self.conf.minimal_instance_type}
    ex = self.assertRaises(exc.HTTPBadRequest, self.client.stacks.create, stack_name=stack_name, parameters=parameters, template=self.unavailable_template)
    self.assertIn('ResourceTypeUnavailable', ex.message.decode('utf-8'))
    self.assertIn('OS::Sahara::NodeGroupTemplate', ex.message.decode('utf-8'))