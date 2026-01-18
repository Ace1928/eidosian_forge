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
def test_scheduled_group_creation(self):
    self.set_http_response(status_code=200)
    self.service_connection.create_scheduled_group_action('foo', 'scheduled-foo', desired_capacity=1, start_time=datetime(2013, 1, 1, 22, 55, 31), end_time=datetime(2013, 2, 1, 22, 55, 31), min_size=1, max_size=2, recurrence='0 10 * * *')
    self.assert_request_parameters({'Action': 'PutScheduledUpdateGroupAction', 'AutoScalingGroupName': 'foo', 'ScheduledActionName': 'scheduled-foo', 'MaxSize': 2, 'MinSize': 1, 'DesiredCapacity': 1, 'EndTime': '2013-02-01T22:55:31', 'StartTime': '2013-01-01T22:55:31', 'Recurrence': '0 10 * * *'}, ignore_params_values=['Version'])