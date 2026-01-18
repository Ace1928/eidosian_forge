from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_get_total(self):
    self.responses.get(self.entity_base, json={'total': 1})
    total = self.manager.total()
    self.assertEqual(1, total)