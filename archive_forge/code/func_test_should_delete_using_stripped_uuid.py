from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
def test_should_delete_using_stripped_uuid(self):
    bad_href = 'http://badsite.com/' + self.entity_id
    self.test_should_delete(bad_href)