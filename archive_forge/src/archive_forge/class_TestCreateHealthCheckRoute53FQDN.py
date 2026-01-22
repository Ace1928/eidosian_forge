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
class TestCreateHealthCheckRoute53FQDN(AWSMockServiceTestCase):
    connection_class = Route53Connection

    def setUp(self):
        super(TestCreateHealthCheckRoute53FQDN, self).setUp()

    def default_body(self):
        return b'\n<CreateHealthCheckResponse xmlns="https://route53.amazonaws.com/doc/2013-04-01/">\n   <HealthCheck>\n      <Id>f9abfe10-8d2a-4bbd-8f35-796f0f8572f2</Id>\n      <CallerReference>3246ac17-b651-4295-a5c8-c132a59693d7</CallerReference>\n      <HealthCheckConfig>\n         <Port>443</Port>\n         <Type>HTTPS</Type>\n         <ResourcePath>/health_check</ResourcePath>\n         <FullyQualifiedDomainName>example.com</FullyQualifiedDomainName>\n         <RequestInterval>30</RequestInterval>\n         <FailureThreshold>3</FailureThreshold>\n      </HealthCheckConfig>\n   </HealthCheck>\n</CreateHealthCheckResponse>\n        '

    def test_create_health_check_fqdn(self):
        self.set_http_response(status_code=201)
        hc = HealthCheck(ip_addr='', port=443, hc_type='HTTPS', resource_path='/health_check', fqdn='example.com')
        hc_xml = hc.to_xml()
        self.assertTrue('<FullyQualifiedDomainName>' in hc_xml)
        self.assertFalse('<IPAddress>' in hc_xml)
        response = self.service_connection.create_health_check(hc)
        hc_resp = response['CreateHealthCheckResponse']['HealthCheck']['HealthCheckConfig']
        self.assertEqual(hc_resp['FullyQualifiedDomainName'], 'example.com')
        self.assertEqual(hc_resp['ResourcePath'], '/health_check')
        self.assertEqual(hc_resp['Type'], 'HTTPS')
        self.assertEqual(hc_resp['Port'], '443')
        self.assertEqual(hc_resp['ResourcePath'], '/health_check')
        self.assertEqual(response['CreateHealthCheckResponse']['HealthCheck']['Id'], 'f9abfe10-8d2a-4bbd-8f35-796f0f8572f2')