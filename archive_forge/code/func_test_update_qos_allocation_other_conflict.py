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
def test_update_qos_allocation_other_conflict(self):
    mock_rsp_get = self._get_allocation_response({RESOURCE_PROVIDER_UUID: {'c': 3}})
    self.placement_fixture.mock_get.side_effect = 10 * [mock_rsp_get]
    mock_rsp_put = mock.Mock()
    mock_rsp_put.text = ''
    mock_rsp_put.json = lambda: {'errors': [{'code': 'some other error code'}]}
    self.placement_fixture.mock_put.side_effect = ks_exc.Conflict(response=mock_rsp_put)
    self.assertRaises(ks_exc.Conflict, self.placement_api_client.update_qos_allocation, consumer_uuid=CONSUMER_UUID, alloc_diff={RESOURCE_PROVIDER_UUID: {}})
    self.placement_fixture.mock_put.assert_called_once()