import unittest
import six
from apitools.base.py import list_pager
from apitools.base.py.testing import mock
from samples.dns_sample.dns_v1 import dns_v1_client
from samples.dns_sample.dns_v1 import dns_v1_messages
class DnsGenClientTest(unittest.TestCase):

    def setUp(self):
        self.mocked_dns_v1 = mock.Client(dns_v1_client.DnsV1)
        self.mocked_dns_v1.Mock()
        self.addCleanup(self.mocked_dns_v1.Unmock)

    def testFlatPath(self):
        get_method_config = self.mocked_dns_v1.projects.GetMethodConfig('Get')
        self.assertIsNone(get_method_config.flat_path)
        self.assertEquals('projects/{project}', get_method_config.relative_path)

    def testRecordSetList(self):
        response_record_set = dns_v1_messages.ResourceRecordSet(kind=u'dns#resourceRecordSet', name=u'zone.com.', rrdatas=[u'1.2.3.4'], ttl=21600, type=u'A')
        self.mocked_dns_v1.resourceRecordSets.List.Expect(dns_v1_messages.DnsResourceRecordSetsListRequest(project=u'my-project', managedZone=u'test_zone_name', type=u'green', maxResults=100), dns_v1_messages.ResourceRecordSetsListResponse(rrsets=[response_record_set]))
        results = list(list_pager.YieldFromList(self.mocked_dns_v1.resourceRecordSets, dns_v1_messages.DnsResourceRecordSetsListRequest(project='my-project', managedZone='test_zone_name', type='green'), limit=100, field='rrsets'))
        self.assertEquals([response_record_set], results)