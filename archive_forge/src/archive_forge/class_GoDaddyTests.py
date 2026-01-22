import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_GODADDY
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.godaddy import GoDaddyDNSDriver
class GoDaddyTests(unittest.TestCase):

    def setUp(self):
        GoDaddyMockHttp.type = None
        GoDaddyDNSDriver.connectionCls.conn_class = GoDaddyMockHttp
        self.driver = GoDaddyDNSDriver(*DNS_PARAMS_GODADDY)

    def assertHasKeys(self, dictionary, keys):
        for key in keys:
            self.assertTrue(key in dictionary, 'key "%s" not in dictionary' % key)

    def test_list_zones(self):
        zones = self.driver.list_zones()
        self.assertEqual(len(zones), 5)
        self.assertEqual(zones[0].id, '177184419')
        self.assertEqual(zones[0].domain, 'aperture-platform.com')

    def test_ex_check_availability(self):
        check = self.driver.ex_check_availability('wazzlewobbleflooble.com')
        self.assertEqual(check.available, True)
        self.assertEqual(check.price, 14.99)

    def test_ex_list_tlds(self):
        tlds = self.driver.ex_list_tlds()
        self.assertEqual(len(tlds), 331)
        self.assertEqual(tlds[0].name, 'academy')
        self.assertEqual(tlds[0].type, 'GENERIC')

    def test_ex_get_purchase_schema(self):
        schema = self.driver.ex_get_purchase_schema('com')
        self.assertEqual(schema['id'], 'https://api.godaddy.com/DomainPurchase#')

    def test_ex_get_agreements(self):
        ags = self.driver.ex_get_agreements('com')
        self.assertEqual(len(ags), 1)
        self.assertEqual(ags[0].title, 'Domain Name Registration Agreement')

    def test_ex_purchase_domain(self):
        fixtures = DNSFileFixtures('godaddy')
        document = fixtures.load('purchase_request.json')
        order = self.driver.ex_purchase_domain(document)
        self.assertEqual(order.order_id, 1)

    def test_list_records(self):
        zone = Zone(id='177184419', domain='aperture-platform.com', type='master', ttl=None, driver=self.driver)
        records = self.driver.list_records(zone)
        self.assertEqual(len(records), 14)
        self.assertEqual(records[0].type, RecordType.A)
        self.assertEqual(records[0].name, '@')
        self.assertEqual(records[0].data, '50.63.202.42')
        self.assertEqual(records[0].id, '@:A')

    def test_get_record(self):
        record = self.driver.get_record('aperture-platform.com', 'www:A')
        self.assertEqual(record.id, 'www:A')
        self.assertEqual(record.name, 'www')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.data, '50.63.202.42')

    def test_create_record(self):
        zone = Zone(id='177184419', domain='aperture-platform.com', type='master', ttl=None, driver=self.driver)
        record = self.driver.create_record(zone=zone, name='www', type=RecordType.A, data='50.63.202.42')
        self.assertEqual(record.id, 'www:A')
        self.assertEqual(record.name, 'www')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.data, '50.63.202.42')

    def test_update_record(self):
        record = self.driver.get_record('aperture-platform.com', 'www:A')
        record = self.driver.update_record(record=record, name='www', type=RecordType.A, data='50.63.202.22')
        self.assertEqual(record.id, 'www:A')
        self.assertEqual(record.name, 'www')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.data, '50.63.202.22')

    def test_get_zone(self):
        zone = self.driver.get_zone('aperture-platform.com')
        self.assertEqual(zone.id, '177184419')
        self.assertEqual(zone.domain, 'aperture-platform.com')

    def test_delete_zone(self):
        zone = Zone(id='177184419', domain='aperture-platform.com', type='master', ttl=None, driver=self.driver)
        self.driver.delete_zone(zone)