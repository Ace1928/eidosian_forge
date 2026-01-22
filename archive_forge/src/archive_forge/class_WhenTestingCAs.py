from oslo_utils import timeutils
from barbicanclient.tests import test_client
from barbicanclient.v1 import cas
class WhenTestingCAs(test_client.BaseEntityResource):

    def setUp(self):
        self._setUp('cas')
        self.ca = CAData()
        self.manager = self.client.cas

    def test_should_get_lazy(self, ca_ref=None):
        ca_ref = ca_ref or self.entity_href
        data = self.ca.get_dict(ca_ref)
        m = self.responses.get(self.entity_href, json=data)
        ca = self.manager.get(ca_ref=ca_ref)
        self.assertIsInstance(ca, cas.CA)
        self.assertEqual(ca_ref, ca._ca_ref)
        self.assertFalse(m.called)
        self.assertEqual(self.ca.plugin_ca_id, ca.plugin_ca_id)
        self.assertEqual(self.entity_href, m.last_request.url)

    def test_should_get_lazy_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self.test_should_get_lazy(bad_href)

    def test_should_get_lazy_using_only_uuid(self):
        self.test_should_get_lazy(self.entity_id)

    def test_should_get_lazy_in_meta(self):
        data = self.ca.get_dict(self.entity_href)
        m = self.responses.get(self.entity_href, json=data)
        ca = self.manager.get(ca_ref=self.entity_href)
        self.assertIsInstance(ca, cas.CA)
        self.assertEqual(self.entity_href, ca._ca_ref)
        self.assertFalse(m.called)
        self.assertEqual(self.ca.name, ca.name)
        self.assertEqual(self.entity_href, m.last_request.url)

    def test_should_get_list(self):
        ca_resp = self.entity_href
        data = {'cas': [ca_resp for v in range(3)]}
        m = self.responses.get(self.entity_base, json=data)
        ca_list = self.manager.list(limit=10, offset=5)
        self.assertTrue(len(ca_list) == 3)
        self.assertIsInstance(ca_list[0], cas.CA)
        self.assertEqual(self.entity_href, ca_list[0].ca_ref)
        self.assertEqual(self.entity_base, m.last_request.url.split('?')[0])
        self.assertEqual(['10'], m.last_request.qs['limit'])
        self.assertEqual(['5'], m.last_request.qs['offset'])

    def test_should_fail_get_invalid_ca(self):
        self.assertRaises(ValueError, self.manager.get, **{'ca_ref': '12345'})

    def test_should_get_ca_that_has_no_meta_description(self):
        self.ca = CAData(description=None)
        data = self.ca.get_dict(self.entity_href)
        m = self.responses.get(self.entity_href, json=data)
        ca = self.manager.get(ca_ref=self.entity_href)
        self.assertIsInstance(ca, cas.CA)
        self.assertEqual(self.entity_href, ca._ca_ref)
        self.assertFalse(m.called)
        self.assertIsNone(self.ca.description)
        self.assertIsNone(ca.description)
        self.assertEqual(self.entity_href, m.last_request.url)

    def test_get_formatted_data(self):
        c_entity = cas.CA(api=None, expiration=self.ca.expiration, plugin_name=self.ca.plugin_name, created=self.ca.created)
        data = c_entity._get_formatted_data()
        self.assertEqual(self.ca.plugin_name, data[6])
        self.assertEqual(timeutils.parse_isotime(self.ca.expiration).isoformat(), data[8])