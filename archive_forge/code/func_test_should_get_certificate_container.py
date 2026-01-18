from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_get_certificate_container(self):
    data = self.container.get_dict(self.entity_href, type='certificate')
    self.responses.get(self.entity_href, json=data)
    container = self.manager.get(container_ref=self.entity_href)
    self.assertIsInstance(container, containers.Container)
    self.assertEqual(self.entity_href, container.container_ref)
    self.assertEqual(self.entity_href, self.responses.last_request.url)
    self.assertIsInstance(container, containers.CertificateContainer)
    self.assertIsNotNone(container.certificate)
    self.assertIsNotNone(container.private_key)
    self.assertIsNotNone(container.private_key_passphrase)
    self.assertIsNotNone(container.intermediates)