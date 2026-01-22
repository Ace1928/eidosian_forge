from tests.compat import mock
import re
import xml.dom.minidom
from boto.exception import BotoServerError
from boto.route53.connection import Route53Connection
from boto.route53.exception import DNSServerError
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets, Record
from boto.route53.zone import Zone
from nose.plugins.attrib import attr
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
@attr(route53=True)
class TestRoute53Connection(AWSMockServiceTestCase):
    connection_class = Route53Connection

    def setUp(self):
        super(TestRoute53Connection, self).setUp()
        self.calls = {'count': 0}

    def default_body(self):
        return b'<Route53Result>\n    <Message>It failed.</Message>\n</Route53Result>\n'

    def test_typical_400(self):
        self.set_http_response(status_code=400, header=[['Code', 'AccessDenied']])
        with self.assertRaises(DNSServerError) as err:
            self.service_connection.get_all_hosted_zones()
        self.assertTrue('It failed.' in str(err.exception))

    def test_retryable_400_prior_request_not_complete(self):
        self.set_http_response(status_code=400, body='<?xml version="1.0"?>\n<ErrorResponse xmlns="https://route53.amazonaws.com/doc/2013-04-01/"><Error><Type>Sender</Type><Code>PriorRequestNotComplete</Code><Message>The request was rejected because Route 53 was still processing a prior request.</Message></Error><RequestId>12d222a0-f3d9-11e4-a611-c321a3a00f9c</RequestId></ErrorResponse>\n')
        self.do_retry_handler()

    def test_retryable_400_throttling(self):
        self.set_http_response(status_code=400, body='<?xml version="1.0"?>\n<ErrorResponse xmlns="https://route53.amazonaws.com/doc/2013-04-01/"><Error><Type>Sender</Type><Code>Throttling</Code><Message>Rate exceeded</Message></Error><RequestId>19d0a9a0-f3d9-11e4-a611-c321a3a00f9c</RequestId></ErrorResponse>\n')
        self.do_retry_handler()

    @mock.patch('time.sleep')
    def do_retry_handler(self, sleep_mock):

        def incr_retry_handler(func):

            def _wrapper(*args, **kwargs):
                self.calls['count'] += 1
                return func(*args, **kwargs)
            return _wrapper
        orig_retry = self.service_connection._retry_handler
        self.service_connection._retry_handler = incr_retry_handler(orig_retry)
        self.assertEqual(self.calls['count'], 0)
        with self.assertRaises(BotoServerError):
            self.service_connection.get_all_hosted_zones()
        self.assertEqual(self.calls['count'], 7)
        self.service_connection._retry_handler = orig_retry

    def test_private_zone_invalid_vpc_400(self):
        self.set_http_response(status_code=400, header=[['Code', 'InvalidVPCId']])
        with self.assertRaises(DNSServerError) as err:
            self.service_connection.create_hosted_zone('example.com.', private_zone=True)
        self.assertTrue('It failed.' in str(err.exception))