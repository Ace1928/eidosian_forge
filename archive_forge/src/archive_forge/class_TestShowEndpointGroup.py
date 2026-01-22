from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import endpoint_group
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestShowEndpointGroup(TestEndpointGroup, common.TestShowVPNaaS):

    def setUp(self):
        super(TestShowEndpointGroup, self).setUp()
        self.networkclient.get_vpn_endpoint_group = mock.Mock(return_value=_endpoint_group)
        self.mocked = self.networkclient.get_vpn_endpoint_group
        self.cmd = endpoint_group.ShowEndpointGroup(self.app, self.namespace)