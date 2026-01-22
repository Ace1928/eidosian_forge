import fixtures
from unittest import mock
from oslo_config import cfg
from stevedore import driver
import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
class TestTransportUrlCustomisation(test_utils.BaseTestCase):

    def setUp(self):
        super(TestTransportUrlCustomisation, self).setUp()

        def transport_url_parse(url):
            return transport.TransportURL.parse(self.conf, url)
        self.url1 = transport_url_parse('fake:/vhost1/localhost:5672/?x=1&y=2&z=3')
        self.url2 = transport_url_parse('fake:/vhost2/localhost:5672/?foo=bar')
        self.url3 = transport_url_parse('fake:/vhost1/localhost:5672/?l=1&l=2&l=3')
        self.url4 = transport_url_parse('fake:/vhost2/localhost:5672/?d=x:1&d=y:2&d=z:3')
        self.url5 = transport_url_parse('fake://noport:/?')

    def test_hash(self):
        urls = {}
        urls[self.url1] = self.url1
        urls[self.url2] = self.url2
        urls[self.url3] = self.url3
        urls[self.url4] = self.url4
        urls[self.url5] = self.url5
        self.assertEqual(3, len(urls))

    def test_eq(self):
        self.assertEqual(self.url1, self.url3)
        self.assertEqual(self.url2, self.url4)
        self.assertNotEqual(self.url1, self.url4)

    def test_query(self):
        self.assertEqual({'x': '1', 'y': '2', 'z': '3'}, self.url1.query)
        self.assertEqual({'foo': 'bar'}, self.url2.query)
        self.assertEqual({'l': '1,2,3'}, self.url3.query)
        self.assertEqual({'d': 'x:1,y:2,z:3'}, self.url4.query)

    def test_noport(self):
        self.assertIsNone(self.url5.hosts[0].port)