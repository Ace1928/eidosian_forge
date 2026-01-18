from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_fail_add_invalid_secret_object(self):
    container = self.manager.create()
    self.assertRaises(ValueError, container.add, 'Not-a-secret', 'Actually a string')