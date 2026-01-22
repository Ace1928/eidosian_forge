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
class TestPutNotificationConfiguration(AWSMockServiceTestCase):
    connection_class = AutoScaleConnection

    def setUp(self):
        super(TestPutNotificationConfiguration, self).setUp()

    def default_body(self):
        return b'\n            <PutNotificationConfigurationResponse>\n              <ResponseMetadata>\n                <RequestId>requestid</RequestId>\n              </ResponseMetadata>\n            </PutNotificationConfigurationResponse>\n        '

    def test_autoscaling_group_put_notification_configuration(self):
        self.set_http_response(status_code=200)
        autoscale = AutoScalingGroup(name='ana', launch_config='lauch_config', min_size=1, max_size=2, termination_policies=['OldestInstance', 'OldestLaunchConfiguration'])
        self.service_connection.put_notification_configuration(autoscale, 'arn:aws:sns:us-east-1:19890506:AutoScaling-Up', ['autoscaling:EC2_INSTANCE_LAUNCH'])
        self.assert_request_parameters({'Action': 'PutNotificationConfiguration', 'AutoScalingGroupName': 'ana', 'NotificationTypes.member.1': 'autoscaling:EC2_INSTANCE_LAUNCH', 'TopicARN': 'arn:aws:sns:us-east-1:19890506:AutoScaling-Up'}, ignore_params_values=['Version'])