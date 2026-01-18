from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import loading as keystone
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from neutron_lib._i18n import _
from neutron_lib.exceptions import placement as n_exc
from neutron_lib import fixture
from neutron_lib.placement import client as place_client
from neutron_lib.tests import _base as base
def test_update_qos_allocation_to_zero(self):
    mock_rsp_get = self._get_allocation_response({RESOURCE_PROVIDER_UUID: {'resources': {'a': 3, 'b': 2}}})
    self.placement_fixture.mock_get.side_effect = [mock_rsp_get]
    self.placement_api_client.update_qos_allocation(consumer_uuid=CONSUMER_UUID, alloc_diff={RESOURCE_PROVIDER_UUID: {'a': -3, 'b': -2}})
    self.placement_fixture.mock_put.assert_called_once_with('/allocations/%s' % CONSUMER_UUID, {'allocations': {}})