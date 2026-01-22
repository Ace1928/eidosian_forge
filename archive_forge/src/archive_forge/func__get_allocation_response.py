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
def _get_allocation_response(self, allocation):
    mock_rsp_get = mock.Mock()
    mock_rsp_get.json = lambda: {'allocations': allocation}
    return mock_rsp_get