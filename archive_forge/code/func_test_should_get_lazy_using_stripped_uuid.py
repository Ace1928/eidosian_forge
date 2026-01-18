from oslo_utils import timeutils
from barbicanclient.tests import test_client
from barbicanclient.v1 import cas
def test_should_get_lazy_using_stripped_uuid(self):
    bad_href = 'http://badsite.com/' + self.entity_id
    self.test_should_get_lazy(bad_href)