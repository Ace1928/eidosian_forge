from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_delete_from_object(self, container_ref=None):
    container_ref = container_ref or self.entity_href
    data = self.container.get_dict(container_ref)
    m = self.responses.get(self.entity_href, json=data)
    n = self.responses.delete(self.entity_href, status_code=204)
    container = self.manager.get(container_ref=container_ref)
    self.assertEqual(container_ref, container.container_ref)
    container.delete()
    self.assertTrue(m.called)
    self.assertTrue(n.called)
    self.assertIsNone(container.container_ref)