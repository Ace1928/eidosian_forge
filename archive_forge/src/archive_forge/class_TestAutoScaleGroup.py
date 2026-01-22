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
class TestAutoScaleGroup(AWSMockServiceTestCase):
    connection_class = AutoScaleConnection

    def setUp(self):
        super(TestAutoScaleGroup, self).setUp()

    def default_body(self):
        return b'\n            <CreateLaunchConfigurationResponse>\n              <ResponseMetadata>\n                <RequestId>requestid</RequestId>\n              </ResponseMetadata>\n            </CreateLaunchConfigurationResponse>\n        '

    def test_autoscaling_group_with_termination_policies(self):
        self.set_http_response(status_code=200)
        autoscale = AutoScalingGroup(name='foo', launch_config='lauch_config', min_size=1, max_size=2, termination_policies=['OldestInstance', 'OldestLaunchConfiguration'], instance_id='test-id')
        self.service_connection.create_auto_scaling_group(autoscale)
        self.assert_request_parameters({'Action': 'CreateAutoScalingGroup', 'AutoScalingGroupName': 'foo', 'LaunchConfigurationName': 'lauch_config', 'MaxSize': 2, 'MinSize': 1, 'TerminationPolicies.member.1': 'OldestInstance', 'TerminationPolicies.member.2': 'OldestLaunchConfiguration', 'InstanceId': 'test-id'}, ignore_params_values=['Version'])

    def test_autoscaling_group_single_vpc_zone_identifier(self):
        self.set_http_response(status_code=200)
        autoscale = AutoScalingGroup(name='foo', vpc_zone_identifier='vpc_zone_1')
        self.service_connection.create_auto_scaling_group(autoscale)
        self.assert_request_parameters({'Action': 'CreateAutoScalingGroup', 'AutoScalingGroupName': 'foo', 'VPCZoneIdentifier': 'vpc_zone_1'}, ignore_params_values=['MaxSize', 'MinSize', 'LaunchConfigurationName', 'Version'])

    def test_autoscaling_group_vpc_zone_identifier_list(self):
        self.set_http_response(status_code=200)
        autoscale = AutoScalingGroup(name='foo', vpc_zone_identifier=['vpc_zone_1', 'vpc_zone_2'])
        self.service_connection.create_auto_scaling_group(autoscale)
        self.assert_request_parameters({'Action': 'CreateAutoScalingGroup', 'AutoScalingGroupName': 'foo', 'VPCZoneIdentifier': 'vpc_zone_1,vpc_zone_2'}, ignore_params_values=['MaxSize', 'MinSize', 'LaunchConfigurationName', 'Version'])

    def test_autoscaling_group_vpc_zone_identifier_multi(self):
        self.set_http_response(status_code=200)
        autoscale = AutoScalingGroup(name='foo', vpc_zone_identifier='vpc_zone_1,vpc_zone_2')
        self.service_connection.create_auto_scaling_group(autoscale)
        self.assert_request_parameters({'Action': 'CreateAutoScalingGroup', 'AutoScalingGroupName': 'foo', 'VPCZoneIdentifier': 'vpc_zone_1,vpc_zone_2'}, ignore_params_values=['MaxSize', 'MinSize', 'LaunchConfigurationName', 'Version'])