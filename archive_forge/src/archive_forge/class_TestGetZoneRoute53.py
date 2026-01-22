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
class TestGetZoneRoute53(AWSMockServiceTestCase):
    connection_class = Route53Connection

    def setUp(self):
        super(TestGetZoneRoute53, self).setUp()

    def default_body(self):
        return b'\n<ListHostedZonesResponse xmlns="https://route53.amazonaws.com/doc/2012-02-29/">\n    <HostedZones>\n        <HostedZone>\n            <Id>/hostedzone/Z1111</Id>\n            <Name>example2.com.</Name>\n            <CallerReference>aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee</CallerReference>\n            <Config/>\n            <ResourceRecordSetCount>3</ResourceRecordSetCount>\n        </HostedZone>\n        <HostedZone>\n            <Id>/hostedzone/Z2222</Id>\n            <Name>example1.com.</Name>\n            <CallerReference>aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeef</CallerReference>\n            <Config/>\n            <ResourceRecordSetCount>6</ResourceRecordSetCount>\n        </HostedZone>\n        <HostedZone>\n            <Id>/hostedzone/Z3333</Id>\n            <Name>example.com.</Name>\n            <CallerReference>aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeeg</CallerReference>\n            <Config/>\n            <ResourceRecordSetCount>6</ResourceRecordSetCount>\n        </HostedZone>\n    </HostedZones>\n    <IsTruncated>false</IsTruncated>\n    <MaxItems>100</MaxItems>\n</ListHostedZonesResponse>\n        '

    def test_list_zones(self):
        self.set_http_response(status_code=201)
        response = self.service_connection.get_all_hosted_zones()
        domains = ['example2.com.', 'example1.com.', 'example.com.']
        print(response['ListHostedZonesResponse']['HostedZones'][0])
        for d in response['ListHostedZonesResponse']['HostedZones']:
            print('Removing: %s' % d['Name'])
            domains.remove(d['Name'])
        self.assertEqual(domains, [])

    def test_get_zone(self):
        self.set_http_response(status_code=201)
        response = self.service_connection.get_zone('example.com.')
        self.assertTrue(isinstance(response, Zone))
        self.assertEqual(response.name, 'example.com.')