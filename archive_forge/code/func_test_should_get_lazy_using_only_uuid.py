from oslo_utils import timeutils
from barbicanclient.tests import test_client
from barbicanclient.v1 import cas
def test_should_get_lazy_using_only_uuid(self):
    self.test_should_get_lazy(self.entity_id)