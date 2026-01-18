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
def test_update_qos_allocation_rp_not_exists(self):
    mock_rsp = mock.Mock()
    mock_rsp.json = lambda: {'allocations': {'other:rp:uuid': {'c': 3}}}
    self.placement_fixture.mock_get.side_effect = [mock_rsp]
    self.assertRaises(n_exc.PlacementAllocationRpNotExists, self.placement_api_client.update_qos_allocation, consumer_uuid=CONSUMER_UUID, alloc_diff={RESOURCE_PROVIDER_UUID: {'a': 1, 'b': 1}})