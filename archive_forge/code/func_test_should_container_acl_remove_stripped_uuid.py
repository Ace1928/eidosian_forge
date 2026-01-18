from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_container_acl_remove_stripped_uuid(self):
    bad_href = 'http://badsite.com/containers/' + self.container_uuid
    self.test_should_container_acl_remove(bad_href)