from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
class TestRDSOptionGroups(AWSMockServiceTestCase):
    connection_class = RDSConnection

    def setUp(self):
        super(TestRDSOptionGroups, self).setUp()

    def default_body(self):
        return '\n        <DescribeOptionGroupsResponse xmlns="http://rds.amazonaws.com/doc/2013-05-15/">\n          <DescribeOptionGroupsResult>\n            <OptionGroupsList>\n              <OptionGroup>\n                <MajorEngineVersion>11.2</MajorEngineVersion>\n                <OptionGroupName>myoptiongroup</OptionGroupName>\n                <EngineName>oracle-se1</EngineName>\n                <OptionGroupDescription>Test option group</OptionGroupDescription>\n                <Options/>\n              </OptionGroup>\n              <OptionGroup>\n                <MajorEngineVersion>11.2</MajorEngineVersion>\n                <OptionGroupName>default:oracle-se1-11-2</OptionGroupName>\n                <EngineName>oracle-se1</EngineName>\n                <OptionGroupDescription>Default Option Group.</OptionGroupDescription>\n                <Options/>\n              </OptionGroup>\n            </OptionGroupsList>\n          </DescribeOptionGroupsResult>\n          <ResponseMetadata>\n            <RequestId>e4b234d9-84d5-11e1-87a6-71059839a52b</RequestId>\n          </ResponseMetadata>\n        </DescribeOptionGroupsResponse>\n        '

    def test_describe_option_groups(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.describe_option_groups()
        self.assertEqual(len(response), 2)
        options = response[0]
        self.assertEqual(options.name, 'myoptiongroup')
        self.assertEqual(options.description, 'Test option group')
        self.assertEqual(options.engine_name, 'oracle-se1')
        self.assertEqual(options.major_engine_version, '11.2')
        options = response[1]
        self.assertEqual(options.name, 'default:oracle-se1-11-2')
        self.assertEqual(options.description, 'Default Option Group.')
        self.assertEqual(options.engine_name, 'oracle-se1')
        self.assertEqual(options.major_engine_version, '11.2')