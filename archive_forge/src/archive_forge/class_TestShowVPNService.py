from unittest import mock
import uuid
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import vpnservice
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestShowVPNService(TestVPNService, common.TestShowVPNaaS):

    def setUp(self):
        super(TestShowVPNService, self).setUp()
        self.networkclient.get_vpn_service = mock.Mock(return_value=_vpnservice)
        self.mocked = self.networkclient.get_vpn_service
        self.cmd = vpnservice.ShowVPNService(self.app, self.namespace)