import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
class TestListInstanceGroups(AWSMockServiceTestCase):
    connection_class = EmrConnection

    def default_body(self):
        return b'\n<ListInstanceGroupsResponse xmlns="http://elasticmapreduce.amazonaws.com/doc/2009-03-31">\n  <ListInstanceGroupsResult>\n    <InstanceGroups>\n      <member>\n        <Id>ig-aaaaaaaaaaaaa</Id>\n        <InstanceType>m1.large</InstanceType>\n        <Market>ON_DEMAND</Market>\n        <Status>\n          <StateChangeReason>\n            <Message>Job flow terminated</Message>\n            <Code>CLUSTER_TERMINATED</Code>\n          </StateChangeReason>\n          <State>TERMINATED</State>\n          <Timeline>\n            <CreationDateTime>2014-01-24T01:21:21Z</CreationDateTime>\n            <ReadyDateTime>2014-01-24T01:25:08Z</ReadyDateTime>\n            <EndDateTime>2014-01-24T02:19:46Z</EndDateTime>\n          </Timeline>\n        </Status>\n        <Name>Master instance group</Name>\n        <RequestedInstanceCount>1</RequestedInstanceCount>\n        <RunningInstanceCount>0</RunningInstanceCount>\n        <InstanceGroupType>MASTER</InstanceGroupType>\n      </member>\n      <member>\n        <Id>ig-aaaaaaaaaaab</Id>\n        <InstanceType>m1.large</InstanceType>\n        <Market>ON_DEMAND</Market>\n        <Status>\n          <StateChangeReason>\n            <Message>Job flow terminated</Message>\n            <Code>CLUSTER_TERMINATED</Code>\n          </StateChangeReason>\n          <State>TERMINATED</State>\n          <Timeline>\n            <CreationDateTime>2014-01-24T01:21:21Z</CreationDateTime>\n            <ReadyDateTime>2014-01-24T01:25:26Z</ReadyDateTime>\n            <EndDateTime>2014-01-24T02:19:46Z</EndDateTime>\n          </Timeline>\n        </Status>\n        <Name>Core instance group</Name>\n        <RequestedInstanceCount>2</RequestedInstanceCount>\n        <RunningInstanceCount>0</RunningInstanceCount>\n        <InstanceGroupType>CORE</InstanceGroupType>\n      </member>\n    </InstanceGroups>\n  </ListInstanceGroupsResult>\n  <ResponseMetadata>\n    <RequestId>aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee</RequestId>\n  </ResponseMetadata>\n</ListInstanceGroupsResponse>\n\n'

    def test_list_instance_groups(self):
        self.set_http_response(200)
        with self.assertRaises(TypeError):
            self.service_connection.list_instance_groups()
        response = self.service_connection.list_instance_groups(cluster_id='j-123')
        self.assert_request_parameters({'Action': 'ListInstanceGroups', 'ClusterId': 'j-123', 'Version': '2009-03-31'})
        self.assertTrue(isinstance(response, InstanceGroupList))
        self.assertEqual(len(response.instancegroups), 2)
        self.assertTrue(isinstance(response.instancegroups[0], InstanceGroupInfo))
        self.assertEqual(response.instancegroups[0].id, 'ig-aaaaaaaaaaaaa')
        self.assertEqual(response.instancegroups[0].instancegrouptype, 'MASTER')
        self.assertEqual(response.instancegroups[0].instancetype, 'm1.large')
        self.assertEqual(response.instancegroups[0].market, 'ON_DEMAND')
        self.assertEqual(response.instancegroups[0].name, 'Master instance group')
        self.assertEqual(response.instancegroups[0].requestedinstancecount, '1')
        self.assertEqual(response.instancegroups[0].runninginstancecount, '0')
        self.assertTrue(isinstance(response.instancegroups[0].status, ClusterStatus))
        self.assertEqual(response.instancegroups[0].status.state, 'TERMINATED')