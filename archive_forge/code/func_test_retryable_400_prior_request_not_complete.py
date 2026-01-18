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
def test_retryable_400_prior_request_not_complete(self):
    self.set_http_response(status_code=400, body='<?xml version="1.0"?>\n<ErrorResponse xmlns="https://route53.amazonaws.com/doc/2013-04-01/"><Error><Type>Sender</Type><Code>PriorRequestNotComplete</Code><Message>The request was rejected because Route 53 was still processing a prior request.</Message></Error><RequestId>12d222a0-f3d9-11e4-a611-c321a3a00f9c</RequestId></ErrorResponse>\n')
    self.do_retry_handler()