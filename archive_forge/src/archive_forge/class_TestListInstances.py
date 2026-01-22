import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
class TestListInstances(AWSMockServiceTestCase):
    connection_class = EmrConnection

    def default_body(self):
        return b'\n<ListInstancesResponse xmlns="http://elasticmapreduce.amazonaws.com/doc/2009-03-31">\n  <ListInstancesResult>\n    <Instances>\n      <member>\n        <Id>ci-123456789abc</Id>\n        <Status>\n          <StateChangeReason>\n            <Message>Cluster was terminated.</Message>\n            <Code>CLUSTER_TERMINATED</Code>\n          </StateChangeReason>\n          <State>TERMINATED</State>\n          <Timeline>\n            <CreationDateTime>2014-01-24T01:21:26Z</CreationDateTime>\n            <ReadyDateTime>2014-01-24T01:25:25Z</ReadyDateTime>\n            <EndDateTime>2014-01-24T02:19:46Z</EndDateTime>\n          </Timeline>\n        </Status>\n        <PrivateDnsName>ip-10-0-0-60.us-west-1.compute.internal</PrivateDnsName>\n        <PublicIpAddress>54.0.0.1</PublicIpAddress>\n        <PublicDnsName>ec2-54-0-0-1.us-west-1.compute.amazonaws.com</PublicDnsName>\n        <Ec2InstanceId>i-aaaaaaaa</Ec2InstanceId>\n        <PrivateIpAddress>10.0.0.60</PrivateIpAddress>\n      </member>\n      <member>\n        <Id>ci-123456789abd</Id>\n        <Status>\n          <StateChangeReason>\n            <Message>Cluster was terminated.</Message>\n            <Code>CLUSTER_TERMINATED</Code>\n          </StateChangeReason>\n          <State>TERMINATED</State>\n          <Timeline>\n            <CreationDateTime>2014-01-24T01:21:26Z</CreationDateTime>\n            <ReadyDateTime>2014-01-24T01:25:25Z</ReadyDateTime>\n            <EndDateTime>2014-01-24T02:19:46Z</EndDateTime>\n          </Timeline>\n        </Status>\n        <PrivateDnsName>ip-10-0-0-61.us-west-1.compute.internal</PrivateDnsName>\n        <PublicIpAddress>54.0.0.2</PublicIpAddress>\n        <PublicDnsName>ec2-54-0-0-2.us-west-1.compute.amazonaws.com</PublicDnsName>\n        <Ec2InstanceId>i-aaaaaaab</Ec2InstanceId>\n        <PrivateIpAddress>10.0.0.61</PrivateIpAddress>\n      </member>\n      <member>\n        <Id>ci-123456789abe3</Id>\n        <Status>\n          <StateChangeReason>\n            <Message>Cluster was terminated.</Message>\n            <Code>CLUSTER_TERMINATED</Code>\n          </StateChangeReason>\n          <State>TERMINATED</State>\n          <Timeline>\n            <CreationDateTime>2014-01-24T01:21:33Z</CreationDateTime>\n            <ReadyDateTime>2014-01-24T01:25:08Z</ReadyDateTime>\n            <EndDateTime>2014-01-24T02:19:46Z</EndDateTime>\n          </Timeline>\n        </Status>\n        <PrivateDnsName>ip-10-0-0-62.us-west-1.compute.internal</PrivateDnsName>\n        <PublicIpAddress>54.0.0.3</PublicIpAddress>\n        <PublicDnsName>ec2-54-0-0-3.us-west-1.compute.amazonaws.com</PublicDnsName>\n        <Ec2InstanceId>i-aaaaaaac</Ec2InstanceId>\n        <PrivateIpAddress>10.0.0.62</PrivateIpAddress>\n      </member>\n    </Instances>\n  </ListInstancesResult>\n  <ResponseMetadata>\n    <RequestId>aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee</RequestId>\n  </ResponseMetadata>\n</ListInstancesResponse>\n        '

    def test_list_instances(self):
        self.set_http_response(200)
        with self.assertRaises(TypeError):
            self.service_connection.list_instances()
        response = self.service_connection.list_instances(cluster_id='j-123')
        self.assertTrue(isinstance(response, InstanceList))
        self.assertEqual(len(response.instances), 3)
        self.assertTrue(isinstance(response.instances[0], InstanceInfo))
        self.assertEqual(response.instances[0].ec2instanceid, 'i-aaaaaaaa')
        self.assertEqual(response.instances[0].id, 'ci-123456789abc')
        self.assertEqual(response.instances[0].privatednsname, 'ip-10-0-0-60.us-west-1.compute.internal')
        self.assertEqual(response.instances[0].privateipaddress, '10.0.0.60')
        self.assertEqual(response.instances[0].publicdnsname, 'ec2-54-0-0-1.us-west-1.compute.amazonaws.com')
        self.assertEqual(response.instances[0].publicipaddress, '54.0.0.1')
        self.assert_request_parameters({'Action': 'ListInstances', 'ClusterId': 'j-123', 'Version': '2009-03-31'})

    def test_list_instances_with_group_id(self):
        self.set_http_response(200)
        response = self.service_connection.list_instances(cluster_id='j-123', instance_group_id='abc')
        self.assert_request_parameters({'Action': 'ListInstances', 'ClusterId': 'j-123', 'InstanceGroupId': 'abc', 'Version': '2009-03-31'})

    def test_list_instances_with_types(self):
        self.set_http_response(200)
        response = self.service_connection.list_instances(cluster_id='j-123', instance_group_types=['MASTER', 'TASK'])
        self.assert_request_parameters({'Action': 'ListInstances', 'ClusterId': 'j-123', 'InstanceGroupTypes.member.1': 'MASTER', 'InstanceGroupTypes.member.2': 'TASK', 'Version': '2009-03-31'})