from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.autoscaling import scaling_policy as aws_sp
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
@mock.patch.object(aws_sp.AWSScalingPolicy, '_get_ec2_signed_url')
def test_scaling_policy_refid_signed_url(self, mock_get_ec2_url):
    t = template_format.parse(as_template)
    stack = utils.parse_stack(t, params=as_params)
    rsrc = self.create_scaling_policy(t, stack, 'WebServerScaleUpPolicy')
    mock_get_ec2_url.return_value = 'http://signed_url'
    self.assertEqual('http://signed_url', rsrc.FnGetRefId())