import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
from openstack import version as openstack_version
@mock.patch.object(cloud_region.CloudRegion, 'get_session')
def test_session_endpoint_not_found(self, mock_get_session):
    exc_to_raise = ksa_exceptions.catalog.EndpointNotFound
    mock_get_session.return_value.get_endpoint.side_effect = exc_to_raise
    cc = cloud_region.CloudRegion('test1', 'region-al', {}, auth_plugin=mock.Mock())
    self.assertIsNone(cc.get_session_endpoint('notfound'))