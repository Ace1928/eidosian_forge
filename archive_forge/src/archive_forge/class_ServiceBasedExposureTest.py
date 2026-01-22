from heatclient import exc
import keystoneclient
from heat_integrationtests.functional import functional_base
class ServiceBasedExposureTest(functional_base.FunctionalTestsBase):
    unavailable_service = 'Sahara'
    unavailable_template = '\nheat_template_version: 2015-10-15\nparameters:\n  instance_type:\n    type: string\nresources:\n  not_available:\n    type: OS::Sahara::NodeGroupTemplate\n    properties:\n      plugin_name: fake\n      hadoop_version: 0.1\n      flavor: {get_param: instance_type}\n      node_processes: []\n'

    def setUp(self):
        super(ServiceBasedExposureTest, self).setUp()
        if self._is_sahara_deployed():
            self.skipTest('Sahara is actually deployed, can not run negative tests on Sahara resources availability.')

    def _is_sahara_deployed(self):
        try:
            self.identity_client.get_endpoint_url('data-processing', self.conf.region)
        except keystoneclient.exceptions.EndpointNotFound:
            return False
        return True

    def test_unavailable_resources_not_listed(self):
        resources = self.client.resource_types.list()
        self.assertFalse(any((self.unavailable_service in r.resource_type for r in resources)))

    def test_unavailable_resources_not_created(self):
        stack_name = self._stack_rand_name()
        parameters = {'instance_type': self.conf.minimal_instance_type}
        ex = self.assertRaises(exc.HTTPBadRequest, self.client.stacks.create, stack_name=stack_name, parameters=parameters, template=self.unavailable_template)
        self.assertIn('ResourceTypeUnavailable', ex.message.decode('utf-8'))
        self.assertIn('OS::Sahara::NodeGroupTemplate', ex.message.decode('utf-8'))