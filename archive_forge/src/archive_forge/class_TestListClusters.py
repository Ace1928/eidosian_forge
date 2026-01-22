import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
class TestListClusters(AWSMockServiceTestCase):
    connection_class = EmrConnection

    def default_body(self):
        return b'\n<ListClustersResponse xmlns="http://elasticmapreduce.amazonaws.com/doc/2009-03-31">\n  <ListClustersResult>\n    <Clusters>\n      <member>\n        <Id>j-aaaaaaaaaaaa</Id>\n        <Status>\n          <StateChangeReason>\n            <Message>Terminated by user request</Message>\n            <Code>USER_REQUEST</Code>\n          </StateChangeReason>\n          <State>TERMINATED</State>\n          <Timeline>\n            <CreationDateTime>2014-01-24T01:21:21Z</CreationDateTime>\n            <ReadyDateTime>2014-01-24T01:25:26Z</ReadyDateTime>\n            <EndDateTime>2014-01-24T02:19:46Z</EndDateTime>\n          </Timeline>\n        </Status>\n        <Name>analytics test</Name>\n        <NormalizedInstanceHours>10</NormalizedInstanceHours>\n      </member>\n      <member>\n        <Id>j-aaaaaaaaaaaab</Id>\n        <Status>\n          <StateChangeReason>\n            <Message>Terminated by user request</Message>\n            <Code>USER_REQUEST</Code>\n          </StateChangeReason>\n          <State>TERMINATED</State>\n          <Timeline>\n            <CreationDateTime>2014-01-21T02:53:08Z</CreationDateTime>\n            <ReadyDateTime>2014-01-21T02:56:40Z</ReadyDateTime>\n            <EndDateTime>2014-01-21T03:40:22Z</EndDateTime>\n          </Timeline>\n        </Status>\n        <Name>test job</Name>\n        <NormalizedInstanceHours>20</NormalizedInstanceHours>\n      </member>\n    </Clusters>\n  </ListClustersResult>\n  <ResponseMetadata>\n    <RequestId>aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee</RequestId>\n  </ResponseMetadata>\n</ListClustersResponse>\n        '

    def test_list_clusters(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.list_clusters()
        self.assert_request_parameters({'Action': 'ListClusters', 'Version': '2009-03-31'})
        self.assertTrue(isinstance(response, ClusterSummaryList))
        self.assertEqual(len(response.clusters), 2)
        self.assertTrue(isinstance(response.clusters[0], ClusterSummary))
        self.assertEqual(response.clusters[0].name, 'analytics test')
        self.assertEqual(response.clusters[0].normalizedinstancehours, '10')
        self.assertTrue(isinstance(response.clusters[0].status, ClusterStatus))
        self.assertEqual(response.clusters[0].status.state, 'TERMINATED')
        self.assertTrue(isinstance(response.clusters[0].status.timeline, ClusterTimeline))
        self.assertEqual(response.clusters[0].status.timeline.creationdatetime, '2014-01-24T01:21:21Z')
        self.assertEqual(response.clusters[0].status.timeline.readydatetime, '2014-01-24T01:25:26Z')
        self.assertEqual(response.clusters[0].status.timeline.enddatetime, '2014-01-24T02:19:46Z')
        self.assertTrue(isinstance(response.clusters[0].status.statechangereason, ClusterStateChangeReason))
        self.assertEqual(response.clusters[0].status.statechangereason.code, 'USER_REQUEST')
        self.assertEqual(response.clusters[0].status.statechangereason.message, 'Terminated by user request')

    def test_list_clusters_created_before(self):
        self.set_http_response(status_code=200)
        date = datetime.now()
        response = self.service_connection.list_clusters(created_before=date)
        self.assert_request_parameters({'Action': 'ListClusters', 'CreatedBefore': date.strftime(boto.utils.ISO8601), 'Version': '2009-03-31'})

    def test_list_clusters_created_after(self):
        self.set_http_response(status_code=200)
        date = datetime.now()
        response = self.service_connection.list_clusters(created_after=date)
        self.assert_request_parameters({'Action': 'ListClusters', 'CreatedAfter': date.strftime(boto.utils.ISO8601), 'Version': '2009-03-31'})

    def test_list_clusters_states(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.list_clusters(cluster_states=['RUNNING', 'WAITING'])
        self.assert_request_parameters({'Action': 'ListClusters', 'ClusterStates.member.1': 'RUNNING', 'ClusterStates.member.2': 'WAITING', 'Version': '2009-03-31'})