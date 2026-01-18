import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def test_update_removal_policies(self):
    rp_template = '\nheat_template_version: 2014-10-16\nresources:\n  random_group:\n    type: OS::Heat::ResourceGroup\n    properties:\n      count: 5\n      removal_policies: []\n      resource_def:\n        type: OS::Heat::RandomString\n'
    stack_identifier = self.stack_create(template=rp_template)
    self.assertEqual({u'random_group': u'OS::Heat::ResourceGroup'}, self.list_resources(stack_identifier))
    group_resources = self.list_group_resources(stack_identifier, 'random_group')
    expected_resources = {u'0': u'OS::Heat::RandomString', u'1': u'OS::Heat::RandomString', u'2': u'OS::Heat::RandomString', u'3': u'OS::Heat::RandomString', u'4': u'OS::Heat::RandomString'}
    self.assertEqual(expected_resources, group_resources)
    update_template = rp_template.replace('removal_policies: []', "removal_policies: [{resource_list: ['1', '2', '3']}]")
    self.update_stack(stack_identifier, update_template)
    group_resources = self.list_group_resources(stack_identifier, 'random_group')
    expected_resources = {u'0': u'OS::Heat::RandomString', u'4': u'OS::Heat::RandomString', u'5': u'OS::Heat::RandomString', u'6': u'OS::Heat::RandomString', u'7': u'OS::Heat::RandomString'}
    self.assertEqual(expected_resources, group_resources)