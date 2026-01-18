from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
def test_should_get_using_only_uuid(self):
    self.test_should_get(self.entity_id)