import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
class TestDescribeJobFlow(DescribeJobFlowsTestBase):

    def test_describe_jobflow(self):
        self.set_http_response(200)
        response = self.service_connection.describe_jobflow('j-aaaaaa')
        self.assertTrue(isinstance(response, JobFlow))
        self.assert_request_parameters({'Action': 'DescribeJobFlows', 'JobFlowIds.member.1': 'j-aaaaaa'}, ignore_params_values=['Version'])