from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_reload_attributes_after_store(self):
    data = {'container_ref': self.entity_href}
    self.responses.post(self.entity_base + '/', json=data)
    data = self.container.get_dict(self.entity_href)
    self.responses.get(self.entity_href, json=data)
    container = self.manager.create(name=self.container.name, secrets=self.container.generic_secrets)
    self.assertIsNone(container.status)
    self.assertIsNone(container.created)
    self.assertIsNone(container.updated)
    container_href = container.store()
    self.assertEqual(self.entity_href, container_href)
    self.assertIsNotNone(container.status)
    self.assertIsNotNone(container.created)