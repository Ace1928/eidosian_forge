from unittest import mock
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_validate_BlockDeviceMappings_without_Ebs_property(self):
    t = template_format.parse(inline_templates.as_template)
    lcp = t['Resources']['LaunchConfig']['Properties']
    bdm = [{'DeviceName': 'vdb'}]
    lcp['BlockDeviceMappings'] = bdm
    stack = utils.parse_stack(t, params=inline_templates.as_params)
    self.stub_ImageConstraint_validate()
    self.stub_FlavorConstraint_validate()
    e = self.assertRaises(exception.StackValidationFailed, self.validate_launch_config, stack)
    self.assertIn('Ebs is missing, this is required', str(e))