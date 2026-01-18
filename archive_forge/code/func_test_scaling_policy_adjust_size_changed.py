from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_scaling_policy_adjust_size_changed(self):
    t = template_format.parse(as_template)
    stack = utils.parse_stack(t, params=as_params)
    up_policy = self.create_scaling_policy(t, stack, 'my-policy')
    group = stack['my-group']
    self.patchobject(group, 'resize')
    self.patchobject(group, '_lb_reload')
    mock_fin_scaling = self.patchobject(group, '_finished_scaling')
    with mock.patch.object(group, '_check_scaling_allowed') as mock_isa:
        self.assertIsNone(up_policy.handle_signal())
        mock_isa.assert_called_once_with(60)
        mock_fin_scaling.assert_called_once_with(60, 'change_in_capacity : 1', size_changed=True)