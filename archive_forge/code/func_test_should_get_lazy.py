from oslo_utils import timeutils
from barbicanclient.tests import test_client
from barbicanclient.v1 import cas
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