import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
class TestCloudFormationListStackResources(CloudFormationConnectionBase):

    def default_body(self):
        return b'\n            <ListStackResourcesResponse>\n              <ListStackResourcesResult>\n                <StackResourceSummaries>\n                  <member>\n                    <ResourceStatus>CREATE_COMPLETE</ResourceStatus>\n                    <LogicalResourceId>SampleDB</LogicalResourceId>\n                    <LastUpdatedTime>2011-06-21T20:25:57Z</LastUpdatedTime>\n                    <PhysicalResourceId>My-db-ycx</PhysicalResourceId>\n                    <ResourceType>AWS::RDS::DBInstance</ResourceType>\n                  </member>\n                  <member>\n                    <ResourceStatus>CREATE_COMPLETE</ResourceStatus>\n                    <LogicalResourceId>CPUAlarmHigh</LogicalResourceId>\n                    <LastUpdatedTime>2011-06-21T20:29:23Z</LastUpdatedTime>\n                    <PhysicalResourceId>MyStack-CPUH-PF</PhysicalResourceId>\n                    <ResourceType>AWS::CloudWatch::Alarm</ResourceType>\n                  </member>\n                </StackResourceSummaries>\n              </ListStackResourcesResult>\n              <ResponseMetadata>\n                <RequestId>2d06e36c-ac1d-11e0-a958-f9382b6eb86b</RequestId>\n              </ResponseMetadata>\n            </ListStackResourcesResponse>\n        '

    def test_list_stack_resources(self):
        self.set_http_response(status_code=200)
        resources = self.service_connection.list_stack_resources('MyStack', next_token='next_token')
        self.assertEqual(len(resources), 2)
        self.assertEqual(resources[0].last_updated_time, datetime(2011, 6, 21, 20, 25, 57))
        self.assertEqual(resources[0].logical_resource_id, 'SampleDB')
        self.assertEqual(resources[0].physical_resource_id, 'My-db-ycx')
        self.assertEqual(resources[0].resource_status, 'CREATE_COMPLETE')
        self.assertEqual(resources[0].resource_status_reason, None)
        self.assertEqual(resources[0].resource_type, 'AWS::RDS::DBInstance')
        self.assertEqual(resources[1].last_updated_time, datetime(2011, 6, 21, 20, 29, 23))
        self.assertEqual(resources[1].logical_resource_id, 'CPUAlarmHigh')
        self.assertEqual(resources[1].physical_resource_id, 'MyStack-CPUH-PF')
        self.assertEqual(resources[1].resource_status, 'CREATE_COMPLETE')
        self.assertEqual(resources[1].resource_status_reason, None)
        self.assertEqual(resources[1].resource_type, 'AWS::CloudWatch::Alarm')
        self.assert_request_parameters({'Action': 'ListStackResources', 'NextToken': 'next_token', 'StackName': 'MyStack', 'Version': '2010-05-15'})