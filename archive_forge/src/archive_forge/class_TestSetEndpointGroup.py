from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import endpoint_group
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestSetEndpointGroup(TestEndpointGroup, common.TestSetVPNaaS):

    def setUp(self):
        super(TestSetEndpointGroup, self).setUp()
        self.networkclient.update_vpn_endpoint_group = mock.Mock(return_value=_endpoint_group)
        self.mocked = self.networkclient.update_vpn_endpoint_group
        self.cmd = endpoint_group.SetEndpointGroup(self.app, self.namespace)