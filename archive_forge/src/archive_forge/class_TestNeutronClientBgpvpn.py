import copy
from unittest import mock
from openstack.network.v2 import bgpvpn as _bgpvpn
from openstack import resource as sdk_resource
from osc_lib.utils import columns as column_util
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.router_association import\
from neutronclient.osc.v2.networking_bgpvpn.router_association import\
from neutronclient.osc.v2.networking_bgpvpn.router_association import\
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
class TestNeutronClientBgpvpn(test_fakes.TestNeutronClientOSCV2):

    def setUp(self):
        super(TestNeutronClientBgpvpn, self).setUp()
        self.neutronclient.find_resource = mock.Mock(side_effect=lambda resource, name_or_id, project_id=None, cmd_resource=None, parent_id=None, fields=None: {'id': name_or_id, 'tenant_id': _FAKE_PROJECT_ID})
        self.neutronclient.find_resource_by_id = mock.Mock(side_effect=lambda resource, resource_id, cmd_resource=None, parent_id=None, fields=None: {'id': resource_id, 'tenant_id': _FAKE_PROJECT_ID})
        nc_osc_utils.find_project = mock.Mock(side_effect=lambda _, name_or_id, __: mock.Mock(id=name_or_id))