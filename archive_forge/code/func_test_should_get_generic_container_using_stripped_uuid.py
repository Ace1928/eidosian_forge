from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_get_generic_container_using_stripped_uuid(self):
    bad_href = 'http://badsite.com/' + self.entity_id
    self.test_should_get_generic_container(bad_href)