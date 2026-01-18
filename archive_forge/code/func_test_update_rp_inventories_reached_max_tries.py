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
def test_update_rp_inventories_reached_max_tries(self):
    mock_resp = mock.Mock()
    mock_resp.text = ''
    mock_resp.json = lambda: {'errors': [{'code': 'placement.concurrent_update'}]}
    self.placement_fixture.mock_put.side_effect = 10 * [ks_exc.Conflict(response=mock_resp)]
    self.assertRaises(n_exc.PlacementResourceProviderGenerationConflict, self.placement_api_client.update_resource_provider_inventories, resource_provider_uuid='resource provider uuid', inventories={}, resource_provider_generation=None)
    self.assertEqual(10, self.placement_fixture.mock_put.call_count)