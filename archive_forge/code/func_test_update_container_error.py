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
def test_update_container_error(self):
    """Swift error re-raised as OSCE"""
    self.register_uris([dict(method='POST', uri=self.container_endpoint, status_code=409)])
    self.assertRaises(exceptions.SDKException, self.cloud.update_container, self.container, dict(foo='bar'))
    self.assert_calls()