from oslo_utils import timeutils
from barbicanclient.tests import test_client
from barbicanclient.v1 import cas
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