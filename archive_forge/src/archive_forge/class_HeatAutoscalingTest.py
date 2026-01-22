from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
class HeatAutoscalingTest(functional_base.FunctionalTestsBase):
    template = '\nheat_template_version: 2014-10-16\n\nresources:\n  random_group:\n    type: OS::Heat::AutoScalingGroup\n    properties:\n      cooldown: 0\n      desired_capacity: 3\n      max_size: 5\n      min_size: 2\n      resource:\n        type: OS::Heat::RandomString\n\n  scale_up_policy:\n    type: OS::Heat::ScalingPolicy\n    properties:\n      adjustment_type: change_in_capacity\n      auto_scaling_group_id: { get_resource: random_group }\n      scaling_adjustment: 1\n\n  scale_down_policy:\n    type: OS::Heat::ScalingPolicy\n    properties:\n      adjustment_type: change_in_capacity\n      auto_scaling_group_id: { get_resource: random_group }\n      scaling_adjustment: -1\n\noutputs:\n  all_values:\n    value: {get_attr: [random_group, outputs_list, value]}\n  value_0:\n    value: {get_attr: [random_group, resource.0.value]}\n  value_1:\n    value: {get_attr: [random_group, resource.1.value]}\n  value_2:\n    value: {get_attr: [random_group, resource.2.value]}\n  asg_size:\n    value: {get_attr: [random_group, current_size]}\n'
    template_nested = '\nheat_template_version: 2014-10-16\n\nresources:\n  random_group:\n    type: OS::Heat::AutoScalingGroup\n    properties:\n      desired_capacity: 3\n      max_size: 5\n      min_size: 2\n      resource:\n        type: randomstr.yaml\n\noutputs:\n  all_values:\n    value: {get_attr: [random_group, outputs_list, random_str]}\n  value_0:\n    value: {get_attr: [random_group, resource.0.random_str]}\n  value_1:\n    value: {get_attr: [random_group, resource.1.random_str]}\n  value_2:\n    value: {get_attr: [random_group, resource.2.random_str]}\n'
    template_randomstr = '\nheat_template_version: 2013-05-23\n\nresources:\n  random_str:\n    type: OS::Heat::RandomString\n\noutputs:\n  random_str:\n    value: {get_attr: [random_str, value]}\n'

    def _assert_output_values(self, stack_id):
        stack = self.client.stacks.get(stack_id)
        all_values = self._stack_output(stack, 'all_values')
        self.assertEqual(3, len(all_values))
        self.assertEqual(all_values[0], self._stack_output(stack, 'value_0'))
        self.assertEqual(all_values[1], self._stack_output(stack, 'value_1'))
        self.assertEqual(all_values[2], self._stack_output(stack, 'value_2'))

    def test_asg_scale_up_max_size(self):
        stack_id = self.stack_create(template=self.template, expected_status='CREATE_COMPLETE')
        stack = self.client.stacks.get(stack_id)
        asg_size = self._stack_output(stack, 'asg_size')
        self.assertEqual(3, asg_size)
        asg = self.client.resources.get(stack_id, 'random_group')
        max_size = 5
        for num in range(asg_size + 1, max_size + 2):
            expected_resources = num if num <= max_size else max_size
            self.client.resources.signal(stack_id, 'scale_up_policy')
            self.assertTrue(test.call_until_true(self.conf.build_timeout, self.conf.build_interval, self.check_autoscale_complete, asg.physical_resource_id, expected_resources, stack_id, 'random_group'))

    def test_asg_scale_down_min_size(self):
        stack_id = self.stack_create(template=self.template, expected_status='CREATE_COMPLETE')
        stack = self.client.stacks.get(stack_id)
        asg_size = self._stack_output(stack, 'asg_size')
        self.assertEqual(3, asg_size)
        asg = self.client.resources.get(stack_id, 'random_group')
        min_size = 2
        for num in range(asg_size - 1, 0, -1):
            expected_resources = num if num >= min_size else min_size
            self.client.resources.signal(stack_id, 'scale_down_policy')
            self.assertTrue(test.call_until_true(self.conf.build_timeout, self.conf.build_interval, self.check_autoscale_complete, asg.physical_resource_id, expected_resources, stack_id, 'random_group'))

    def test_asg_cooldown(self):
        cooldown_tmpl = self.template.replace('cooldown: 0', 'cooldown: 60')
        stack_id = self.stack_create(template=cooldown_tmpl, expected_status='CREATE_COMPLETE')
        stack = self.client.stacks.get(stack_id)
        asg_size = self._stack_output(stack, 'asg_size')
        self.assertEqual(3, asg_size)
        asg = self.client.resources.get(stack_id, 'random_group')
        expected_resources = 3
        self.client.resources.signal(stack_id, 'scale_up_policy')
        self.assertTrue(test.call_until_true(self.conf.build_timeout, self.conf.build_interval, self.check_autoscale_complete, asg.physical_resource_id, expected_resources, stack_id, 'random_group'))

    def test_path_attrs(self):
        stack_id = self.stack_create(template=self.template)
        expected_resources = {'random_group': 'OS::Heat::AutoScalingGroup', 'scale_up_policy': 'OS::Heat::ScalingPolicy', 'scale_down_policy': 'OS::Heat::ScalingPolicy'}
        self.assertEqual(expected_resources, self.list_resources(stack_id))
        self._assert_output_values(stack_id)

    def test_path_attrs_nested(self):
        files = {'randomstr.yaml': self.template_randomstr}
        stack_id = self.stack_create(template=self.template_nested, files=files)
        expected_resources = {'random_group': 'OS::Heat::AutoScalingGroup'}
        self.assertEqual(expected_resources, self.list_resources(stack_id))
        self._assert_output_values(stack_id)