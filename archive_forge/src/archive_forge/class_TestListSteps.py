import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
class TestListSteps(AWSMockServiceTestCase):
    connection_class = EmrConnection

    def default_body(self):
        return b'<ListStepsResponse xmlns="http://elasticmapreduce.amazonaws.com/doc/2009-03-31">\n  <ListStepsResult>\n    <Steps>\n      <member>\n        <Id>abc123</Id>\n        <Status>\n          <StateChangeReason/>\n          <Timeline>\n            <CreationDateTime>2014-07-01T00:00:00.000Z</CreationDateTime>\n          </Timeline>\n          <State>PENDING</State>\n        </Status>\n        <Name>Step 1</Name>\n        <Config>\n          <Jar>/home/hadoop/lib/emr-s3distcp-1.0.jar</Jar>\n          <Args>\n            <member>--src</member>\n            <member>hdfs:///data/test/</member>\n            <member>--dest</member>\n            <member>s3n://test/data</member>\n          </Args>\n          <Properties/>\n        </Config>\n        <ActionOnFailure>CONTINUE</ActionOnFailure>\n      </member>\n      <member>\n        <Id>def456</Id>\n        <Status>\n          <StateChangeReason/>\n          <Timeline>\n            <CreationDateTime>2014-07-01T00:00:00.000Z</CreationDateTime>\n          </Timeline>\n          <State>COMPLETED</State>\n        </Status>\n        <Name>Step 2</Name>\n        <Config>\n          <MainClass>my.main.SomeClass</MainClass>\n          <Jar>s3n://test/jars/foo.jar</Jar>\n        </Config>\n        <ActionOnFailure>CONTINUE</ActionOnFailure>\n      </member>\n      <member>\n        <Id>ghi789</Id>\n        <Status>\n          <StateChangeReason/>\n          <Timeline>\n            <CreationDateTime>2014-07-01T00:00:00.000Z</CreationDateTime>\n          </Timeline>\n          <State>FAILED</State>\n        </Status>\n        <Name>Step 3</Name>\n        <Config>\n          <Jar>s3n://test/jars/bar.jar</Jar>\n          <Args>\n            <member>-arg</member>\n            <member>value</member>\n          </Args>\n          <Properties/>\n        </Config>\n        <ActionOnFailure>TERMINATE_CLUSTER</ActionOnFailure>\n      </member>\n    </Steps>\n  </ListStepsResult>\n  <ResponseMetadata>\n    <RequestId>eff31ee5-0342-11e4-b3c7-9de5a93f6fcb</RequestId>\n  </ResponseMetadata>\n</ListStepsResponse>\n'

    def test_list_steps(self):
        self.set_http_response(200)
        with self.assertRaises(TypeError):
            self.service_connection.list_steps()
        response = self.service_connection.list_steps(cluster_id='j-123')
        self.assert_request_parameters({'Action': 'ListSteps', 'ClusterId': 'j-123', 'Version': '2009-03-31'})
        self.assertTrue(isinstance(response, StepSummaryList))
        self.assertEqual(response.steps[0].name, 'Step 1')
        valid_states = ['PENDING', 'RUNNING', 'COMPLETED', 'CANCELLED', 'FAILED', 'INTERRUPTED']
        for step in response.steps:
            self.assertIn(step.status.state, valid_states)
        step = response.steps[0]
        self.assertEqual(step.config.jar, '/home/hadoop/lib/emr-s3distcp-1.0.jar')
        self.assertEqual(len(step.config.args), 4)
        self.assertEqual(step.config.args[0].value, '--src')
        self.assertEqual(step.config.args[1].value, 'hdfs:///data/test/')
        step = response.steps[1]
        self.assertEqual(step.config.mainclass, 'my.main.SomeClass')

    def test_list_steps_with_states(self):
        self.set_http_response(200)
        response = self.service_connection.list_steps(cluster_id='j-123', step_states=['COMPLETED', 'FAILED'])
        self.assert_request_parameters({'Action': 'ListSteps', 'ClusterId': 'j-123', 'StepStates.member.1': 'COMPLETED', 'StepStates.member.2': 'FAILED', 'Version': '2009-03-31'})
        self.assertTrue(isinstance(response, StepSummaryList))
        self.assertEqual(response.steps[0].name, 'Step 1')