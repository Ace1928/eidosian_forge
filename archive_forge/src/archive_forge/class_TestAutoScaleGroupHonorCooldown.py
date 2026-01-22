import base64
from datetime import datetime
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.autoscale import AutoScaleConnection
from boto.ec2.autoscale.group import AutoScalingGroup
from boto.ec2.autoscale.policy import ScalingPolicy
from boto.ec2.autoscale.tag import Tag
from boto.ec2.blockdevicemapping import EBSBlockDeviceType, BlockDeviceMapping
from boto.ec2.autoscale import launchconfig, LaunchConfiguration
class TestAutoScaleGroupHonorCooldown(AWSMockServiceTestCase):
    connection_class = AutoScaleConnection

    def default_body(self):
        return b'\n            <SetDesiredCapacityResponse>\n              <ResponseMetadata>\n                <RequestId>9fb7e2db-6998-11e2-a985-57c82EXAMPLE</RequestId>\n              </ResponseMetadata>\n            </SetDesiredCapacityResponse>\n        '

    def test_honor_cooldown(self):
        self.set_http_response(status_code=200)
        self.service_connection.set_desired_capacity('foo', 10, True)
        self.assert_request_parameters({'Action': 'SetDesiredCapacity', 'AutoScalingGroupName': 'foo', 'DesiredCapacity': 10, 'HonorCooldown': 'true'}, ignore_params_values=['Version'])