from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_store_after_delete_from_object(self):
    data = self.container.get_dict(self.entity_href)
    self.responses.get(self.entity_href, json=data)
    data = self.container.get_dict(self.entity_href)
    self.responses.post(self.entity_base + '/', json=data)
    m = self.responses.delete(self.entity_href, status_code=204)
    container = self.manager.get(container_ref=self.entity_href)
    self.assertIsNotNone(container.container_ref)
    container.delete()
    self.assertEqual(self.entity_href, m.last_request.url)
    self.assertIsNone(container.container_ref)
    container.store()
    self.assertIsNotNone(container.container_ref)