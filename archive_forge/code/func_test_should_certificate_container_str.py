from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_certificate_container_str(self):
    container_obj = self.manager.create_certificate(name=self.container.name)
    self.assertIn(self.container.name, str(container_obj))
    self.assertIn(' certificate ', str(container_obj))