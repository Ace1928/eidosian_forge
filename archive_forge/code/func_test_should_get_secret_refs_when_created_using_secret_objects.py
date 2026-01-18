from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_get_secret_refs_when_created_using_secret_objects(self):
    data = {'container_ref': self.entity_href}
    self.responses.post(self.entity_base + '/', json=data)
    container = self.manager.create(name=self.container.name, secrets=self.container.generic_secrets)
    self.assertEqual(self.container.generic_secret_refs, container.secret_refs)