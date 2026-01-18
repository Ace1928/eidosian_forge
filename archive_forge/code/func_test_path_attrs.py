from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_path_attrs(self):
    stack_id = self.stack_create(template=self.template)
    expected_resources = {'random_group': 'OS::Heat::AutoScalingGroup', 'scale_up_policy': 'OS::Heat::ScalingPolicy', 'scale_down_policy': 'OS::Heat::ScalingPolicy'}
    self.assertEqual(expected_resources, self.list_resources(stack_id))
    self._assert_output_values(stack_id)