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
class TestLaunchConfigurationDescribe(AWSMockServiceTestCase):
    connection_class = AutoScaleConnection

    def default_body(self):
        return b'\n        <DescribeLaunchConfigurationsResponse>\n          <DescribeLaunchConfigurationsResult>\n            <LaunchConfigurations>\n              <member>\n                <AssociatePublicIpAddress>true</AssociatePublicIpAddress>\n                <SecurityGroups/>\n                <CreatedTime>2013-01-21T23:04:42.200Z</CreatedTime>\n                <KernelId/>\n                <LaunchConfigurationName>my-test-lc</LaunchConfigurationName>\n                <UserData/>\n                <InstanceType>m1.small</InstanceType>\n                <LaunchConfigurationARN>arn:aws:autoscaling:us-east-1:803981987763:launchConfiguration:9dbbbf87-6141-428a-a409-0752edbe6cad:launchConfigurationName/my-test-lc</LaunchConfigurationARN>\n                <BlockDeviceMappings/>\n                <ImageId>ami-514ac838</ImageId>\n                <KeyName/>\n                <RamdiskId/>\n                <InstanceMonitoring>\n                  <Enabled>true</Enabled>\n                </InstanceMonitoring>\n                <EbsOptimized>false</EbsOptimized>\n                <ClassicLinkVPCId>vpc-12345</ClassicLinkVPCId>\n                <ClassicLinkVPCSecurityGroups>\n                    <member>sg-1234</member>\n                </ClassicLinkVPCSecurityGroups>\n              </member>\n            </LaunchConfigurations>\n          </DescribeLaunchConfigurationsResult>\n          <ResponseMetadata>\n            <RequestId>d05a22f8-b690-11e2-bf8e-2113fEXAMPLE</RequestId>\n          </ResponseMetadata>\n        </DescribeLaunchConfigurationsResponse>\n        '

    def test_get_all_launch_configurations(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_all_launch_configurations()
        self.assertTrue(isinstance(response, list))
        self.assertEqual(len(response), 1)
        self.assertTrue(isinstance(response[0], LaunchConfiguration))
        self.assertEqual(response[0].associate_public_ip_address, True)
        self.assertEqual(response[0].name, 'my-test-lc')
        self.assertEqual(response[0].instance_type, 'm1.small')
        self.assertEqual(response[0].launch_configuration_arn, 'arn:aws:autoscaling:us-east-1:803981987763:launchConfiguration:9dbbbf87-6141-428a-a409-0752edbe6cad:launchConfigurationName/my-test-lc')
        self.assertEqual(response[0].image_id, 'ami-514ac838')
        self.assertTrue(isinstance(response[0].instance_monitoring, launchconfig.InstanceMonitoring))
        self.assertEqual(response[0].instance_monitoring.enabled, 'true')
        self.assertEqual(response[0].ebs_optimized, False)
        self.assertEqual(response[0].block_device_mappings, [])
        self.assertEqual(response[0].classic_link_vpc_id, 'vpc-12345')
        self.assertEqual(response[0].classic_link_vpc_security_groups, ['sg-1234'])
        self.assert_request_parameters({'Action': 'DescribeLaunchConfigurations'}, ignore_params_values=['Version'])

    def test_get_all_configuration_limited(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_all_launch_configurations(max_records=10, names=['my-test1', 'my-test2'])
        self.assert_request_parameters({'Action': 'DescribeLaunchConfigurations', 'MaxRecords': 10, 'LaunchConfigurationNames.member.1': 'my-test1', 'LaunchConfigurationNames.member.2': 'my-test2'}, ignore_params_values=['Version'])