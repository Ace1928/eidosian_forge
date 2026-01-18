import copy
import json
from testtools import matchers
from heat_integrationtests.functional import functional_base
def test_size_updates_work(self):
    files = {'provider.yaml': self.instance_template}
    env = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 2, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
    stack_identifier = self.stack_create(template=self.template, files=files, environment=env)
    stack = self.client.stacks.get(stack_identifier)
    self.assert_instance_count(stack, 2)
    env2 = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 5, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
    self.update_stack(stack_identifier, self.template, environment=env2, files=files)
    stack = self.client.stacks.get(stack_identifier)
    self.assert_instance_count(stack, 5)