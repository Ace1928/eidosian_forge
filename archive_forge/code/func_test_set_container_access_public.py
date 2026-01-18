import tempfile
from unittest import mock
import testtools
import openstack.cloud.openstackcloud as oc_oc
from openstack import exceptions
from openstack.object_store.v1 import _proxy
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit import base
from openstack import utils
def test_set_container_access_public(self):
    self.register_uris([dict(method='POST', uri=self.container_endpoint, status_code=204, validate=dict(headers={'x-container-read': oc_oc.OBJECT_CONTAINER_ACLS['public']}))])
    self.cloud.set_container_access(self.container, 'public')
    self.assert_calls()