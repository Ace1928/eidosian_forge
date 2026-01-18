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
def test_update_rp_inventory_callee_handles_generation_conflict(self):
    mock_resp = mock.Mock()
    mock_resp.text = ''
    mock_resp.json = lambda: {'errors': [{'code': 'placement.concurrent_update'}]}
    self.placement_fixture.mock_put.side_effect = [ks_exc.Conflict(response=mock_resp), mock.Mock()]
    self.placement_api_client.update_resource_provider_inventory(resource_provider_uuid='resource provider uuid', inventory={}, resource_class='a resource class', resource_provider_generation=None)
    self.assertEqual(2, self.placement_fixture.mock_put.call_count)